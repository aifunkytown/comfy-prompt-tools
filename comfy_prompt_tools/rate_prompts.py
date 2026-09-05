"""
Reads a prompt CSV and asks a local Ollama model to content-rate each row on
a movie-style scale (G, PG, PG-13, R, X, XXX, or REVIEW for suspected
underage content - see rate_prompts.json for the full rubric), writing the
result into "Content Rating" and "Rating Reason" columns in the same CSV.

Rates the "Cleaned Prompt" column when present and non-empty, falling back to
"Positive Prompt" otherwise - matching what actually gets sent to ComfyUI.

Rows already carrying a real Content Rating are skipped on a re-run (so it's
safe to stop and resume, or to run again after adding new rows to a CSV) -
pass --overwrite to force everything to be re-rated instead. Rows whose
Content Rating is "UNPARSED" (the model's response didn't parse into a valid
rating) are always retried on a re-run, --overwrite or not, since UNPARSED
isn't a real rating to begin with.

A model occasionally leaks its base model's built-in safety refusal through
despite being an "abliterated"/uncensored fine-tune (observed verbatim from
gemma4-heretic: "I cannot fulfill this request. I am prohibited from
generating or rewriting content that contains sexually explicit material,
including descriptions of genitalia and pornographic acts."). This is
retried once immediately; if it persists, the row is marked "ERROR" instead
of "UNPARSED" - unlike UNPARSED, ERROR is permanent and is never retried by
this script again, --overwrite included, since a genuine refusal won't be
fixed by trying again later. A row whose Cleaned Prompt is itself "ERROR"
(clean_prompts.py hit the same refusal) is marked ERROR here too without
spending an Ollama call on it.

The rubric lives in rate_prompts.json (checked in, edit it there rather than
in code); rate_prompts.local.json next to it can append personal instructions
via a "system_prompt_addendum" string, same pattern as clean_prompts.py.

Requires Ollama running locally (ollama serve) with the model already pulled:
    ollama pull gemma4:12b

Usage:
    python rate_prompts.py <path-to-csv>

    # Process every *.csv file in the current directory instead of one file:
    python rate_prompts.py

    # Use a different local Ollama model instead of the default:
    python rate_prompts.py <path-to-csv> --model llama3.1:8b

    # Reprocess rows that already have a Content Rating:
    python rate_prompts.py <path-to-csv> --overwrite
"""

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # "localhost" can resolve to ::1 first and get refused if Ollama only binds IPv4
MODEL = "gemma4:12b"
RATING_COLUMN = "Content Rating"
REASON_COLUMN = "Rating Reason"

VALID_RATINGS = ("G", "PG", "PG-13", "R", "X", "XXX", "REVIEW")
RATING_ALIASES = {r.replace("-", ""): r for r in VALID_RATINGS}

try:
    from local_config import (  # run directly: python rate_prompts.py
        MAX_RESPONSE_TOKENS, REQUEST_TIMEOUT_SECONDS, is_refusal_response,
        load_text, load_local_text, sanitize_ascii, strip_thinking,
    )
except ImportError:
    from comfy_prompt_tools.local_config import (  # imported as a package
        MAX_RESPONSE_TOKENS, REQUEST_TIMEOUT_SECONDS, is_refusal_response,
        load_text, load_local_text, sanitize_ascii, strip_thinking,
    )

# rate_prompts.json (checked in) holds "system_prompt_base" - edit the rubric
# there, not in code. rate_prompts.local.json next to it (gitignored, same
# base+local pattern as everywhere else in this project) can add personal
# instructions on top via a "system_prompt_addendum" string.
SYSTEM_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "rate_prompts.json"


def build_system_prompt():
    base = load_text(SYSTEM_PROMPT_CONFIG_PATH, "system_prompt_base")
    addendum = load_local_text(SYSTEM_PROMPT_CONFIG_PATH, "system_prompt_addendum")
    return base + " " + addendum if addendum else base


SYSTEM_PROMPT = build_system_prompt()


def check_ollama_running(timeout=5):
    """True if Ollama's HTTP API is reachable - checked via /api/tags rather
    than /api/generate, since listing local models doesn't require an
    actual (slower, model-dependent) inference request just to prove the
    server is up."""
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def parse_rating_response(text: str):
    """Split the model's 'RATING | reason' response into (rating, reason).
    First strips a reasoning-model <think> block (see
    local_config.strip_thinking) - beyond keeping raw reasoning out of the
    reason field, this matters for correctness here specifically: a
    reasoning model often muses over several candidate ratings by name
    before settling on its actual answer, and since every '|'-separated
    segment gets scanned below, an unstripped think block risks matching
    one of those musings instead of the real, final rating. Some models
    also echo the literal word "RATING" as a label before the actual grade
    instead of substituting it (e.g. "RATING | X | reason", or truncated
    mid-response as just "RATING | X" with no reason at all) - scanning
    every '|'-separated segment for the first one that's a valid rating
    (rather than assuming it's always the first segment) handles that
    label-echo case for free, while an unlabeled "X | reason" still
    matches on its first segment exactly as before. Falls back to
    rating="UNPARSED" (with the full response as the reason, so nothing is
    silently lost) if no segment normalizes to one of VALID_RATINGS (case,
    spacing, and hyphen-insensitive). The reason is free-form model text
    just like a cleaned prompt, and needs the same sanitize_ascii backstop
    - left as-is, a smart quote or similar in it crashes the moment a
    Windows cp1252 console tries to print it."""
    text = strip_thinking(text.strip()).strip()
    parts = text.split("|")
    for i, part in enumerate(parts):
        normalized = part.strip().upper().replace(" ", "").replace(".", "").replace("-", "")
        rating = RATING_ALIASES.get(normalized)
        if rating is not None:
            reason = sanitize_ascii("|".join(parts[i + 1:]).strip())
            return rating, reason
    return "UNPARSED", sanitize_ascii(text)


def get_input_text(row) -> str:
    cleaned = (row.get("Cleaned Prompt") or "").strip()
    return cleaned or (row.get("Positive Prompt") or "").strip()


def _post_ollama(payload):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["response"]


def _rate_prompt_once(text: str, model: str):
    payload = {
        "model": model,
        "prompt": text,
        "system": SYSTEM_PROMPT,
        "stream": False,
        # Some models (e.g. gemma4-heretic) support a hidden "thinking" pass
        # before the visible answer. Left on, they can burn the entire
        # num_predict budget on invisible reasoning and return a completely
        # empty response with done_reason "length" instead of a rating -
        # same failure mode generate_prompt_variations.py/clean_prompts.py
        # already disable this for. strip_thinking() in parse_rating_response
        # above is a second line of defense for a model that ignores this.
        "think": False,
        "options": {"num_predict": MAX_RESPONSE_TOKENS},
    }
    raw_response = _post_ollama(payload)
    if is_refusal_response(raw_response):
        # The base model's safety filter leaking through an "abliterated"
        # fine-tune (see local_config.is_refusal_response) - a resampled
        # request often gets past it, but this is a distinct, permanent
        # failure mode from a merely-malformed response, so it gets exactly
        # one retry here rather than rate_prompt()'s full UNPARSED budget.
        print("  refusal detected, retrying once...", file=sys.stderr)
        raw_response = _post_ollama(payload)
        if is_refusal_response(raw_response):
            return "ERROR", "Model refused twice (safety filter leaking through)"
    return parse_rating_response(raw_response)


def rate_prompt(text: str, model: str = MODEL, attempts: int = 3):
    """Rate text, retrying if the model's response doesn't parse into one of
    VALID_RATINGS. Some models (gemma4-heretic in particular) intermittently
    echo the literal word "RATING" as a label without ever substituting the
    actual grade (e.g. "RATING | reason" instead of "X | reason") - a
    resampled request often gets a well-formed response on the next try, so
    it's worth a few attempts before falling back to UNPARSED for good.
    A rating of "ERROR" (see _rate_prompt_once) is a permanent failure, not
    a transient one - it's returned immediately, without spending the rest
    of the UNPARSED retry budget on a refusal that already survived its own
    dedicated retry."""
    rating, reason = "UNPARSED", ""
    for attempt in range(1, attempts + 1):
        rating, reason = _rate_prompt_once(text, model)
        if rating == "ERROR":
            return rating, reason
        if rating != "UNPARSED":
            return rating, reason
        if attempt < attempts:
            print(f"  UNPARSED response, retrying ({attempt}/{attempts})...", file=sys.stderr)
    return rating, reason


def process_csv(csv_path, args):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "Positive Prompt" not in fieldnames and "Cleaned Prompt" not in fieldnames:
        print(f"Skipping {csv_path}: no 'Positive Prompt' or 'Cleaned Prompt' column found")
        return

    for column in (RATING_COLUMN, REASON_COLUMN):
        if column not in fieldnames:
            fieldnames.append(column)

    tmp_path = str(csv_path) + ".tmp"
    total = len(rows)
    wrote_checkpoint = False
    succeeded = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        existing_rating = row.get(RATING_COLUMN, "").strip()
        # ERROR is permanent (a refusal that survived its own dedicated
        # retry in _rate_prompt_once) - never retried, --overwrite or not.
        if existing_rating == "ERROR":
            continue
        # UNPARSED isn't a real rating, just a record that the model's
        # response didn't parse - always worth another shot on a re-run,
        # same as rate_prompt()'s own in-request retries above, without
        # requiring --overwrite and re-doing every already-successful row.
        if existing_rating and existing_rating != "UNPARSED" and not args.overwrite:
            continue  # already rated, e.g. resuming a previous run

        text = get_input_text(row)
        if not text:
            row[RATING_COLUMN] = ""
            row[REASON_COLUMN] = ""
            continue
        if text == "ERROR":
            # clean_prompts.py's own permanent refusal marker - there's no
            # real text here to rate, and it's not worth an Ollama call.
            row[RATING_COLUMN] = "ERROR"
            row[REASON_COLUMN] = "Cleaned Prompt is a permanent ERROR marker - nothing to rate"
            continue

        print(f"[{i}/{total}] {row.get('File Name', '')}")
        try:
            rating, reason = rate_prompt(text, model=args.model)
            row[RATING_COLUMN] = rating
            row[REASON_COLUMN] = reason
            print(f"  {rating} - {reason}")
            succeeded += 1
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            row[RATING_COLUMN] = ""
            row[REASON_COLUMN] = ""
            failed += 1

        # checkpoint after every row so a crash doesn't lose progress
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        wrote_checkpoint = True

    if wrote_checkpoint:
        os.replace(tmp_path, csv_path)
        print(f"Done. {succeeded} rated, {failed} failed.")
        if failed and not succeeded:
            print(
                f"Warning: every attempted row failed - check that Ollama is running "
                f"({OLLAMA_URL}) and that model '{args.model}' is pulled.",
                file=sys.stderr,
            )
    else:
        print("Nothing to do - every row already has a Content Rating (use --overwrite to redo them).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to the CSV file to process (default: every *.csv file in the current directory)")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model to use (default: {MODEL})")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows that already have a Content Rating (default: skip them)")
    args = parser.parse_args()

    if args.csv_path:
        csv_paths = [args.csv_path]
    else:
        csv_paths = sorted(str(p) for p in Path.cwd().glob("*.csv"))
        if not csv_paths:
            print("No CSV files found in the current directory.", file=sys.stderr)
            sys.exit(1)

    for csv_path in csv_paths:
        print(f"=== {csv_path} ===")
        process_csv(csv_path, args)


if __name__ == "__main__":
    main()
