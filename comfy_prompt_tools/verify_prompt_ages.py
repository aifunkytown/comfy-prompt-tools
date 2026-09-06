"""
Second-pass safety net for clean_prompts.py's age requirement: reads a
prompt CSV's "Cleaned Prompt" column and asks a local Ollama model to check
that every person/humanoid subject it describes has an explicit numeric age
of at least 20 stated (35 if described as mature/older) - the same policy
clean_prompts.py's own system prompts already ask for up front. If every
subject already has one, the text comes back unchanged; if any subject is
missing an age (or has one under 20), the model rewrites just enough to add
or correct it, leaving everything else about the description untouched.

This exists to backfill CSVs cleaned before that policy existed, and to
catch a model occasionally not following its own system prompt - not to
replace clean_prompts.py's own age instructions, which still do the real
work on a normal cleaning pass.

A row is skipped without ever calling the LLM if a local pre-filter is
confident it already meets the requirement (see already_meets_age_
requirement()): the scene reads as a single subject and an explicit age
of at least 20 is already found in the text. Deliberately conservative -
a multi-subject scene, or a scene with no confidently-recognized age
phrasing, always goes through the real LLM check, since a false positive
here would silently skip verifying a row that actually needs it.

Marks each row's "Age Verified" column "yes" once checked (change or not,
pre-filtered or not) - skipped on a re-run unless --overwrite, same
convention as rate_prompts.py.
A model occasionally leaks the base model's built-in safety refusal through
despite being an "abliterated"/uncensored fine-tune (see
local_config.is_refusal_response) - retried once immediately, and marked
the permanent "ERROR" (never retried again, --overwrite included) if it
persists, same as clean_prompts.py/rate_prompts.py. A row whose Cleaned
Prompt is itself blank or the "ERROR" marker is skipped outright - there's
nothing to verify.

Requires Ollama running locally (ollama serve) with the model already pulled.

Usage:
    python verify_prompt_ages.py <path-to-csv>

    # Process every *.csv file in the current directory instead of one file:
    python verify_prompt_ages.py

    # Use a different local Ollama model instead of the default:
    python verify_prompt_ages.py <path-to-csv> --model llama3.1:8b

    # Reprocess rows that already have Age Verified = yes:
    python verify_prompt_ages.py <path-to-csv> --overwrite
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # "localhost" can resolve to ::1 first and get refused if Ollama only binds IPv4
MODEL = "gemma4:12b"
CLEANED_COLUMN = "Cleaned Prompt"
STATUS_COLUMN = "Age Verified"
NOTE_COLUMN = "Age Verification Note"

try:
    from local_config import (  # run directly: python verify_prompt_ages.py
        MAX_RESPONSE_TOKENS, REQUEST_TIMEOUT_SECONDS, is_refusal_response,
        load_text, load_local_text, sanitize_ascii, strip_thinking,
    )
    import sort_prompts_by_category as _spc
except ImportError:
    from comfy_prompt_tools.local_config import (  # imported as a package
        MAX_RESPONSE_TOKENS, REQUEST_TIMEOUT_SECONDS, is_refusal_response,
        load_text, load_local_text, sanitize_ascii, strip_thinking,
    )
    from comfy_prompt_tools import sort_prompts_by_category as _spc

# Local pre-filter so an already-compliant row can skip the LLM call
# entirely (see already_meets_age_requirement()) - deliberately
# conservative, since a false positive here (deciding a row is fine when
# it isn't) would silently let a non-compliant row through completely
# unverified, with no LLM call to ever catch it. A false negative (not
# recognizing a valid age phrasing) just falls through to the normal LLM
# check, so it costs time, never correctness.
_AGE_YEARS_OLD_RE = re.compile(r"\b(\d{1,3})[\s-]*(?:years?|yrs?)[\s-]*old\b", re.IGNORECASE)
_AGE_AGED_RE = re.compile(r"\baged\s+(\d{1,3})\b", re.IGNORECASE)
_AGE_DECADE_WORD_RE = re.compile(
    r"\bin (?:her|his|their) (?:early|mid|late)?\s*(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b",
    re.IGNORECASE,
)
_AGE_DECADE_NUMERIC_RE = re.compile(
    r"\bin (?:her|his|their) (?:early|mid|late)?\s*(20s|30s|40s|50s|60s|70s|80s|90s)\b",
    re.IGNORECASE,
)
_DECADE_WORD_MIN_AGE = {
    "twenties": 20, "thirties": 30, "forties": 40, "fifties": 50,
    "sixties": 60, "seventies": 70, "eighties": 80, "nineties": 90,
}
_DECADE_NUMERIC_MIN_AGE = {f"{n}0s": n * 10 for n in range(2, 10)}


def _find_explicit_ages(text):
    """Every explicit age statement found in text, as the minimum age each
    implies (e.g. "35-year-old" -> 35, "in her late 20s" -> 20) - empty if
    none found. See already_meets_age_requirement() for how this is used;
    on its own this is just detection, not a verdict."""
    ages = [int(m.group(1)) for m in _AGE_YEARS_OLD_RE.finditer(text)]
    ages += [int(m.group(1)) for m in _AGE_AGED_RE.finditer(text)]
    ages += [_DECADE_WORD_MIN_AGE[m.group(1).lower()] for m in _AGE_DECADE_WORD_RE.finditer(text)]
    ages += [_DECADE_NUMERIC_MIN_AGE[m.group(1).lower()] for m in _AGE_DECADE_NUMERIC_RE.finditer(text)]
    return ages


def already_meets_age_requirement(text):
    """True only when it's safe to skip the LLM call for this row entirely:
    the scene reads as a single subject (sort_prompts_by_category.is_group()
    - the same solo/group heuristic already used to sort these CSVs, so a
    multi-subject scene always falls through to the LLM, since confirming
    EVERY subject has an age needs real judgment, not just "found a
    number") AND at least one explicit age was found, with every match
    found (there's normally just one) at least 20."""
    if _spc.is_group({"Cleaned Prompt": text}):
        return False
    ages = _find_explicit_ages(text)
    return bool(ages) and all(age >= 20 for age in ages)

# verify_prompt_ages.json (checked in) holds "system_prompt_base" - edit the
# instructions there, not in code. verify_prompt_ages.local.json next to it
# (gitignored, same base+local pattern as everywhere else in this project)
# can add personal instructions on top via a "system_prompt_addendum" string.
SYSTEM_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "verify_prompt_ages.json"


def build_system_prompt():
    base = load_text(SYSTEM_PROMPT_CONFIG_PATH, "system_prompt_base")
    addendum = load_local_text(SYSTEM_PROMPT_CONFIG_PATH, "system_prompt_addendum")
    return base + " " + addendum if addendum else base


SYSTEM_PROMPT = build_system_prompt()


def check_ollama_running(timeout=5):
    """True if Ollama's HTTP API is reachable - see rate_prompts.py's
    identical helper for the full rationale."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _post_ollama(payload):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["response"]


def verify_prompt_age(text: str, model: str = MODEL):
    """Returns (result_text, status): status is "yes" (result_text is the
    model's - possibly unchanged - description, sanitized and with any
    reasoning-model <think> block stripped) or "ERROR" (result_text is a
    short explanation) if the model refused twice in a row - see the
    module docstring."""
    payload = {
        "model": model,
        "prompt": text,
        "system": SYSTEM_PROMPT,
        "stream": False,
        # Some models (e.g. gemma4-heretic) support a hidden "thinking" pass
        # before the visible answer - see clean_prompts.py's identical
        # comment for the full failure mode this avoids.
        "think": False,
        "options": {"num_predict": MAX_RESPONSE_TOKENS},
    }
    raw_response = _post_ollama(payload)
    if is_refusal_response(raw_response):
        print("  refusal detected, retrying once...", file=sys.stderr)
        raw_response = _post_ollama(payload)
        if is_refusal_response(raw_response):
            return "Model refused twice (safety filter leaking through)", "ERROR"
    result_text = sanitize_ascii(strip_thinking(raw_response).strip())
    return result_text, "yes"


def process_csv(csv_path, args):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if CLEANED_COLUMN not in fieldnames:
        print(f"Skipping {csv_path}: no '{CLEANED_COLUMN}' column found")
        return

    for column in (STATUS_COLUMN, NOTE_COLUMN):
        if column not in fieldnames:
            fieldnames.append(column)

    tmp_path = str(csv_path) + ".tmp"
    total = len(rows)
    wrote_checkpoint = False
    succeeded = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        existing_status = row.get(STATUS_COLUMN, "").strip()
        # ERROR is permanent (a refusal that survived its own dedicated
        # retry in verify_prompt_age) - never retried, --overwrite or not,
        # same as every other script's ERROR rows.
        if existing_status == "ERROR":
            continue
        if existing_status and not args.overwrite:
            continue  # already verified, e.g. resuming a previous run

        text = (row.get(CLEANED_COLUMN) or "").strip()
        if not text or text == "ERROR":
            # Nothing to verify - either never cleaned, or clean_prompts.py's
            # own permanent refusal marker.
            continue

        print(f"[{i}/{total}] {row.get('File Name', '')}")

        if already_meets_age_requirement(text):
            row[STATUS_COLUMN] = "yes"
            row[NOTE_COLUMN] = "already has an explicit age >=20 (pre-filter, no LLM call needed)"
            print(f"  {row[NOTE_COLUMN]}")
            succeeded += 1
        else:
            try:
                result_text, status = verify_prompt_age(text, model=args.model)
            except Exception as e:
                print(f"  error: {e}", file=sys.stderr)
                failed += 1
                continue

            if status == "ERROR":
                row[STATUS_COLUMN] = "ERROR"
                row[NOTE_COLUMN] = result_text
                print(f"  ERROR - {result_text}")
                failed += 1
            else:
                changed = result_text != text
                row[CLEANED_COLUMN] = result_text
                row[STATUS_COLUMN] = "yes"
                row[NOTE_COLUMN] = "age added/corrected" if changed else "no change needed"
                print(f"  {row[NOTE_COLUMN]}")
                succeeded += 1

        # checkpoint after every row so a crash doesn't lose progress
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        wrote_checkpoint = True

    if wrote_checkpoint:
        os.replace(tmp_path, csv_path)
        print(f"Done. {succeeded} verified, {failed} failed.")
        if failed and not succeeded:
            print(
                f"Warning: every attempted row failed - check that Ollama is running "
                f"({OLLAMA_URL}) and that model '{args.model}' is pulled.",
                file=sys.stderr,
            )
    else:
        print("Nothing to do - every row already verified or has nothing to verify (use --overwrite to redo them).")


def verify_all(csv_paths, model=MODEL, overwrite=False):
    """Core logic behind main() - process an explicit list of CSV paths
    without going through argparse. Callable directly by other scripts
    (e.g. clean_and_rate.py)."""
    args = argparse.Namespace(model=model, overwrite=overwrite)
    for csv_path in csv_paths:
        print(f"=== {csv_path} ===")
        process_csv(csv_path, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to the CSV file to process (default: every *.csv file in the current directory)")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model to use (default: {MODEL})")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows that already have Age Verified = yes (default: skip them)")
    args = parser.parse_args()

    if args.csv_path:
        csv_paths = [args.csv_path]
    else:
        csv_paths = sorted(str(p) for p in Path.cwd().glob("*.csv"))
        if not csv_paths:
            print("No CSV files found in the current directory.", file=sys.stderr)
            sys.exit(1)

    verify_all(csv_paths, model=args.model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
