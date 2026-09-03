"""
Reads Positive Prompt column from a ComfyUI prompt-log CSV, asks a local
Ollama model to rewrite each one as a natural-language prompt suitable for
a modern text-to-image model (e.g. Krea 2), and writes the result back into
a "Cleaned Prompt" column in the same CSV.

Each row is also content-rated in that same Ollama call, using
rate_prompts.py's rubric (rate_prompts.json) - one round trip produces both
the rewritten prompt and its "Content Rating"/"Rating Reason" instead of a
separate rate_prompts.py pass making a second call for the same text. A CSV
cleaned before this existed (or one whose Cleaned Prompt was hand-edited
afterward) has no rating yet and won't get one just by re-running this
script without --overwrite (the row's already "done") - run rate_prompts.py
directly on it instead to backfill ratings without re-cleaning.

A row with no Positive Prompt text at all (extract_image_prompts.py writes
one of these for an image with no embedded metadata, instead of skipping
it - see that script's own docstring) falls back to describing the image
directly: a vision-capable Ollama model (VISION_MODEL) is shown the image
itself, via that row's "File Path" column, instead of having no text to
rewrite. This only ever triggers when there's truly no prompt text to work
with - a row that already has one always goes through the normal text
rewrite above, on --model or the default, unchanged. This path is rated in
the same call too.

By default the final saved CSV is trimmed down to just the "Positive Prompt",
"Cleaned Prompt", "Content Rating", and "Rating Reason" columns, dropping
File Name/File Path/Negative Prompt/etc. Pass -v/--verbose to keep every
original column instead. Either way, in-progress checkpoints during a run
always keep every column, so a crash mid-run never loses data for rows not
yet processed.

Requires Ollama running locally (ollama serve) with both models pulled:
    ollama pull gemma4:12b
    ollama pull huihui_ai/qwen2.5-vl-abliterated:7b

Usage:
    python clean_prompts.py <path-to-csv>

    # Keep every original column in the output instead of trimming to just
    # Positive Prompt + Cleaned Prompt:
    python clean_prompts.py <path-to-csv> --verbose

    # Process every *.csv file in the current directory instead of one file:
    python clean_prompts.py

    # Use a different local Ollama model instead of the default (only
    # affects rows that already have prompt text - see VISION_MODEL above
    # for the no-metadata image-description fallback):
    python clean_prompts.py <path-to-csv> --model llama3.1:8b

    # Also submit each cleaned prompt to ComfyUI for rendering as it's cleaned.
    # Keyword-matched LoRAs are turned on automatically, same as
    # rerun_prompts_comfyui.py (see its LORA_RULES):
    python clean_prompts.py <path-to-csv> --submit-to-comfyui --workflow <workflow.json>
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"  # "localhost" can resolve to ::1 first and get refused if Ollama only binds IPv4
MODEL = "gemma4:12b"
# Vision-capable model used only as a fallback, when a row has no prompt
# text at all to rewrite (see the module docstring) - independent of
# --model/MODEL above, which only ever applies to the text-rewrite path.
VISION_MODEL = "huihui_ai/qwen2.5-vl-abliterated:7b"
PROMPT_COLUMN = "Positive Prompt"
OUTPUT_COLUMN = "Cleaned Prompt"
IMAGE_PATH_COLUMN = "File Path"
# Written into PROMPT_COLUMN once the image-description fallback has run for
# a row, in place of leaving it blank - marks that row as "described from
# the image, not from prompt text" so a later re-run (or rerun_prompts_
# comfyui.py reading the same CSV) recognizes it and doesn't try to feed
# this literal string through the text-rewrite model as if it were a real
# prompt.
NOT_FOUND_MARKER = "not found"

RERUN_SCRIPT_DIR = str(Path(__file__).resolve().parent)  # rerun_prompts_comfyui.py lives alongside this script
DEFAULT_WORKFLOW = r"F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json"
DEFAULT_COMFYUI_SERVER = "http://127.0.0.1:8000"

try:
    from local_config import load_local_text, load_text  # run directly: python clean_prompts.py
    import rate_prompts
except ImportError:
    from comfy_prompt_tools.local_config import load_local_text, load_text  # imported as a package
    from comfy_prompt_tools import rate_prompts

RATING_COLUMN = rate_prompts.RATING_COLUMN
REASON_COLUMN = rate_prompts.REASON_COLUMN
# Separates the rewritten/described text from its content rating within one
# combined Ollama response (see _combined_system_prompt/_split_response_and_
# rating below) - distinctive enough that it won't appear in an actual
# prompt or description by accident.
RATING_DELIMITER = "===RATING==="

# clean_prompts.json (checked in) holds "system_prompt_base" - edit the
# prompt there, not in code. clean_prompts.local.json next to it (gitignored,
# same base+local pattern as everywhere else in this project) can add
# personal instructions on top via a "system_prompt_addendum" string.
SYSTEM_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "clean_prompts.json"


def build_system_prompt():
    """The base prompt (clean_prompts.json) is SFW - the age-safety clause
    in it stays baked in unconditionally, a guardrail rather than something
    droppable. clean_prompts.local.json's addendum, if present, is appended
    as-is."""
    base = load_text(SYSTEM_PROMPT_CONFIG_PATH, "system_prompt_base")
    addendum = load_local_text(SYSTEM_PROMPT_CONFIG_PATH, "system_prompt_addendum")
    return base + " " + addendum if addendum else base


def _combined_system_prompt(base_prompt: str) -> str:
    """Appends rate_prompts.py's content-rating rubric onto a rewrite/
    description system prompt, so one Ollama call produces both instead of
    rate_prompts.py needing a separate call over the same text afterward.
    The rubric's own "respond with exactly one line" instruction still
    applies - just to the second part of this combined response, after the
    delimiter, rather than to the whole reply."""
    return (
        f"{base_prompt}\n\n"
        f"After writing your response above, add a new line containing "
        f"exactly \"{RATING_DELIMITER}\" and nothing else, then rate that "
        f"same response using the rubric below, following its own output "
        f"format exactly on the line after the delimiter.\n\n"
        f"{rate_prompts.SYSTEM_PROMPT}"
    )


SYSTEM_PROMPT = _combined_system_prompt(build_system_prompt())
IMAGE_DESCRIPTION_SYSTEM_PROMPT = _combined_system_prompt(
    load_text(SYSTEM_PROMPT_CONFIG_PATH, "image_description_system_prompt")
)

# Curly/smart-typography characters the model sometimes emits despite being
# told not to - mapped to their plain-ASCII equivalents so a bad response
# never makes it into the CSV even if the system prompt is ignored.
_ASCII_REPLACEMENTS = {
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-",                   # en dash
    "—": " - ",                 # em dash
    "…": "...",                 # ellipsis
    " ": " ",                   # non-breaking space
}


def sanitize_ascii(text: str) -> str:
    """Force text down to plain ASCII: map common smart-typography characters
    to their ASCII equivalents, decompose accented letters to their base form
    (e.g. "e" -> "e"), and drop anything left that still isn't ASCII (emoji,
    CJK, etc). Applied as a backstop after every model response, regardless
    of what the system prompt asked for."""
    if not text:
        return text
    for bad, good in _ASCII_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


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


def _split_response_and_rating(raw_response: str):
    """Splits a combined rewrite/description + rating response (see
    _combined_system_prompt) on RATING_DELIMITER into (main_text, rating,
    reason). If the model ignored the delimiter instruction, the whole
    response is kept as main_text and the rating comes back "UNPARSED" -
    getting the rewritten/described text right matters more than the
    rating succeeding, so a rating-parsing miss never costs the row its
    actual prompt text."""
    if RATING_DELIMITER in raw_response:
        main_part, rating_part = raw_response.split(RATING_DELIMITER, 1)
    else:
        main_part, rating_part = raw_response, ""

    main_text = sanitize_ascii(main_part.strip())
    if rating_part.strip():
        rating, reason = rate_prompts.parse_rating_response(rating_part)
    else:
        rating, reason = "UNPARSED", "no rating delimiter found in response"
    return main_text, rating, reason


def clean_prompt(positive_prompt: str, model: str = MODEL):
    """Returns (cleaned_text, rating, reason) - see _split_response_and_rating."""
    payload = {
        "model": model,
        "prompt": positive_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return _split_response_and_rating(result["response"])


def describe_image(image_path, model: str = VISION_MODEL):
    """Ask a local vision-capable Ollama model to describe an image
    directly - the fallback for a row with no prompt text at all to
    rewrite (see the module docstring). Returns (description, rating,
    reason), same as clean_prompt(). image_path is read and sent as
    base64, same request shape as clean_prompt() otherwise. A larger
    timeout than clean_prompt()'s, since a vision model's first pass over
    an image can take noticeably longer than a pure text rewrite."""
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "prompt": "Describe this image.",
        "system": IMAGE_DESCRIPTION_SYSTEM_PROMPT,
        "images": [image_b64],
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return _split_response_and_rating(result["response"])


def submit_cleaned_prompt(rerun, workflow_bundle, server, client_id, row, cleaned_text, randomize_seed):
    """Queue one cleaned prompt on ComfyUI, reusing rerun_prompts_comfyui's workflow
    builder and its keyword-based LoRA toggling."""
    template, positive_id, negative_id, save_ids, lora_node_id = workflow_bundle
    negative_text = (row.get("Negative Prompt") or "").strip()
    name = row.get("File Name") or "row"
    prefix = f"cleaned_{Path(name).stem}"

    lora_matches = rerun.select_loras(cleaned_text)
    lora_summary = ", ".join(f"{lora}@{strength}" for lora, strength in lora_matches) or "none"

    wf = rerun.build_workflow_for_row(
        template, positive_id, negative_id, save_ids,
        cleaned_text, negative_text, prefix, randomize_seed,
        lora_node_id, lora_matches,
    )
    try:
        result = rerun.queue_prompt(server, wf, client_id)
    except urllib.error.URLError as e:
        print(f"  comfyui submit failed: {e}", file=sys.stderr)
        return

    node_errors = result.get("node_errors")
    if node_errors:
        print(f"  comfyui node errors: {node_errors}", file=sys.stderr)
    else:
        print(f"  queued in ComfyUI as prompt_id={result.get('prompt_id')} (LoRAs: {lora_summary})")


def process_csv(csv_path, args, rerun, workflow_bundle, client_id):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if PROMPT_COLUMN not in fieldnames:
        print(f"Skipping {csv_path}: no '{PROMPT_COLUMN}' column found")
        return

    for column in (OUTPUT_COLUMN, RATING_COLUMN, REASON_COLUMN):
        if column not in fieldnames:
            fieldnames.append(column)

    tmp_path = csv_path + ".tmp"
    total = len(rows)
    wrote_checkpoint = False
    succeeded = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        if row.get(OUTPUT_COLUMN, "").strip() and not args.overwrite:
            continue  # already done, e.g. resuming a previous run

        positive = (row.get(PROMPT_COLUMN, "") or "").strip()
        image_path = row.get(IMAGE_PATH_COLUMN, "") or ""
        # NOT_FOUND_MARKER means a prior run already tried the text prompt
        # and found none - treat it the same as genuinely empty, not as
        # real prompt text to run through the text-rewrite model again.
        use_image_fallback = not positive or positive == NOT_FOUND_MARKER

        if use_image_fallback and not (image_path and Path(image_path).is_file()):
            row[OUTPUT_COLUMN] = ""
            continue

        print(f"[{i}/{total}] {row.get('File Name', '')}")
        try:
            if use_image_fallback:
                row[OUTPUT_COLUMN], row[RATING_COLUMN], row[REASON_COLUMN] = describe_image(image_path)
                row[PROMPT_COLUMN] = NOT_FOUND_MARKER
            else:
                row[OUTPUT_COLUMN], row[RATING_COLUMN], row[REASON_COLUMN] = clean_prompt(positive, model=args.model)
            print(f"  Rating: {row[RATING_COLUMN]} - {row[REASON_COLUMN]}")
            succeeded += 1
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            row[OUTPUT_COLUMN] = ""
            row[RATING_COLUMN] = ""
            row[REASON_COLUMN] = ""
            failed += 1

        if args.submit_to_comfyui and row[OUTPUT_COLUMN]:
            submit_cleaned_prompt(
                rerun, workflow_bundle,
                args.server, client_id, row, row[OUTPUT_COLUMN], args.random_seed,
            )

        # checkpoint after every row so a crash doesn't lose progress
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        wrote_checkpoint = True

    if wrote_checkpoint:
        os.replace(tmp_path, csv_path)

        if not args.verbose:
            # Checkpoints above always keep every column (so a crash mid-run doesn't
            # lose File Name/Negative Prompt/etc for rows not yet processed) - only
            # the final saved file gets trimmed down to the prompt + rating columns.
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=[PROMPT_COLUMN, OUTPUT_COLUMN, RATING_COLUMN, REASON_COLUMN], extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)

        print(f"Done. {succeeded} cleaned, {failed} failed.")
        if failed and not succeeded:
            print(
                f"Warning: every attempted row failed - check that Ollama is running "
                f"({OLLAMA_URL}) and that model '{args.model}' (and, for any image-only "
                f"row, '{VISION_MODEL}') is pulled.",
                file=sys.stderr,
            )
    else:
        print("Nothing to do - every row already has a Cleaned Prompt (use --overwrite to redo them).")


def clean_all(csv_paths, submit_to_comfyui=False, workflow=DEFAULT_WORKFLOW, server=DEFAULT_COMFYUI_SERVER,
              random_seed=False, overwrite=False, verbose=False, model=MODEL):
    """Core logic behind main() - process an explicit list of CSV paths
    without going through argparse. Callable directly by other scripts
    (e.g. extract_and_clean.py)."""
    args = argparse.Namespace(
        submit_to_comfyui=submit_to_comfyui, workflow=workflow, server=server,
        random_seed=random_seed, overwrite=overwrite, verbose=verbose, model=model,
    )

    rerun = workflow_bundle = client_id = None
    if submit_to_comfyui:
        sys.path.insert(0, RERUN_SCRIPT_DIR)
        import rerun_prompts_comfyui as rerun

        workflow_path = Path(workflow).expanduser().resolve()
        workflow_bundle = rerun.load_workflow_bundle(workflow_path, server)
        client_id = str(uuid.uuid4())

    for csv_path in csv_paths:
        print(f"=== {csv_path} ===")
        process_csv(csv_path, args, rerun, workflow_bundle, client_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to the CSV file to process (default: every *.csv file in the current directory)")
    parser.add_argument("--submit-to-comfyui", action="store_true",
                         help="After cleaning each row, submit the cleaned prompt to a running ComfyUI server via rerun_prompts_comfyui.py")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW,
                         help=f"Workflow JSON used for every row (default: {DEFAULT_WORKFLOW})")
    parser.add_argument("--server", default=DEFAULT_COMFYUI_SERVER,
                         help=f"ComfyUI server URL for --submit-to-comfyui (default: {DEFAULT_COMFYUI_SERVER})")
    parser.add_argument("--random-seed", action="store_true",
                         help="Randomize seed/noise_seed inputs when submitting to ComfyUI")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows that already have a Cleaned Prompt value (default: skip them)")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model to use (default: {MODEL})")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="Keep every original column in the saved CSV (default: trim the final output down to just "
                              f"'{PROMPT_COLUMN}', '{OUTPUT_COLUMN}', '{RATING_COLUMN}', and '{REASON_COLUMN}')")
    args = parser.parse_args()

    if args.csv_path:
        csv_paths = [args.csv_path]
    else:
        csv_paths = sorted(str(p) for p in Path.cwd().glob("*.csv"))
        if not csv_paths:
            print("No CSV files found in the current directory.", file=sys.stderr)
            sys.exit(1)

    clean_all(
        csv_paths,
        submit_to_comfyui=args.submit_to_comfyui, workflow=args.workflow, server=args.server,
        random_seed=args.random_seed, overwrite=args.overwrite, verbose=args.verbose, model=args.model,
    )


if __name__ == "__main__":
    main()
