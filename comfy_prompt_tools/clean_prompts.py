"""
Reads Positive Prompt column from a ComfyUI prompt-log CSV, asks a local
Ollama model to rewrite each one as a natural-language prompt suitable for
a modern text-to-image model (e.g. Krea 2), and writes the result back into
a "Cleaned Prompt" column in the same CSV.

By default the final saved CSV is trimmed down to just the "Positive Prompt"
and "Cleaned Prompt" columns, dropping File Name/File Path/Negative Prompt/
etc. Pass -v/--verbose to keep every original column instead. Either way,
in-progress checkpoints during a run always keep every column, so a crash
mid-run never loses data for rows not yet processed.

Requires Ollama running locally (ollama serve) with the model already pulled:
    ollama pull gemma4:12b

Usage:
    python clean_prompts.py <path-to-csv>

    # Keep every original column in the output instead of trimming to just
    # Positive Prompt + Cleaned Prompt:
    python clean_prompts.py <path-to-csv> --verbose

    # Process every *.csv file in the current directory instead of one file:
    python clean_prompts.py

    # Also submit each cleaned prompt to ComfyUI for rendering as it's cleaned.
    # Keyword-matched LoRAs are turned on automatically, same as
    # rerun_prompts_comfyui.py (see its LORA_RULES):
    python clean_prompts.py <path-to-csv> --submit-to-comfyui --workflow <workflow.json>
"""

import argparse
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
PROMPT_COLUMN = "Positive Prompt"
OUTPUT_COLUMN = "Cleaned Prompt"

RERUN_SCRIPT_DIR = str(Path(__file__).resolve().parent)  # rerun_prompts_comfyui.py lives alongside this script
DEFAULT_WORKFLOW = r"F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json"
DEFAULT_COMFYUI_SERVER = "http://127.0.0.1:8000"

try:
    from local_config import load_local_text  # run directly: python clean_prompts.py
except ImportError:
    from comfy_prompt_tools.local_config import load_local_text  # imported as a package

# Nominal path used only to derive clean_prompts.local.json's name/location -
# there's no base JSON to read here, just an optional local addendum.
LOCAL_CONFIG_PATH = Path(__file__).resolve().parent / "clean_prompts.json"

SYSTEM_PROMPT_BASE = (
    "You rewrite Stable Diffusion style tag-soup prompts into a single natural-language "
    "sentence or short paragraph describing the image, suitable for a modern text-to-image "
    "model like Krea 2. Keep all important visual details (subject, pose, style, lighting, "
    "colors, setting) but drop artist names, LoRA/embedding tokens, weight syntax like "
    "(word:1.2), and quality boilerplate (masterpiece, best quality, etc). Output only the "
    "rewritten prompt, no preamble or explanation. There should be no drawn, anime, painting, "
    "shading, or otherwise non-realistic words. Wild colors and designs are ok so long as they have realistic people. "
    "If the image is explicit (sex or nudity) and an age is not specified please add that they are 20 years old. Anything marked as mature or older should be set to 35."
    "Descriptions should be detailed include words like realistic photo, high resolution, detailed, and include camera settings. "
    "Adding details is ok to flesh out a picture. It should produce a cinematic image."
    "Also if any character names are given keep those but make sure to add a realistic depiction of before their name. "
    "If there are multiple people in the image specify that their faces should have variety and not be the same."
    "prompts should be between 150 and 1000 characters. "
    "Write in plain ASCII English only - use straight quotes and apostrophes "
    "(' and \"), not curly/smart ones, use a regular hyphen instead of an em "
    "dash or en dash, and never use emoji or non-English characters/scripts, "
    "even for character names (transliterate them to plain ASCII letters)."
)


def build_system_prompt():
    """The base prompt is SFW (the age-safety clause above stays baked in
    unconditionally - it's a guardrail, not something that should be
    droppable). An optional clean_prompts.local.json next to this script
    (gitignored) can add personal instructions - e.g. explicit-content
    direction - via a "system_prompt_addendum" string, appended as-is."""
    addendum = load_local_text(LOCAL_CONFIG_PATH, "system_prompt_addendum")
    return SYSTEM_PROMPT_BASE + " " + addendum if addendum else SYSTEM_PROMPT_BASE


SYSTEM_PROMPT = build_system_prompt()

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


def clean_prompt(positive_prompt: str) -> str:
    payload = {
        "model": MODEL,
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
    return sanitize_ascii(result["response"].strip())


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

    if OUTPUT_COLUMN not in fieldnames:
        fieldnames.append(OUTPUT_COLUMN)

    tmp_path = csv_path + ".tmp"
    total = len(rows)
    wrote_checkpoint = False
    succeeded = 0
    failed = 0

    for i, row in enumerate(rows, 1):
        if row.get(OUTPUT_COLUMN, "").strip() and not args.overwrite:
            continue  # already done, e.g. resuming a previous run

        positive = row.get(PROMPT_COLUMN, "") or ""
        if not positive.strip():
            row[OUTPUT_COLUMN] = ""
            continue

        print(f"[{i}/{total}] {row.get('File Name', '')}")
        try:
            row[OUTPUT_COLUMN] = clean_prompt(positive)
            succeeded += 1
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            row[OUTPUT_COLUMN] = ""
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
            # the final saved file gets trimmed down to just the two prompt columns.
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[PROMPT_COLUMN, OUTPUT_COLUMN], extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        print(f"Done. {succeeded} cleaned, {failed} failed.")
        if failed and not succeeded:
            print(
                f"Warning: every attempted row failed - check that Ollama is running "
                f"({OLLAMA_URL}) and that model '{MODEL}' is pulled.",
                file=sys.stderr,
            )
    else:
        print("Nothing to do - every row already has a Cleaned Prompt (use --overwrite to redo them).")


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
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="Keep every original column in the saved CSV (default: trim the final output down to just "
                              f"'{PROMPT_COLUMN}' and '{OUTPUT_COLUMN}')")
    args = parser.parse_args()

    if args.csv_path:
        csv_paths = [args.csv_path]
    else:
        csv_paths = sorted(str(p) for p in Path.cwd().glob("*.csv"))
        if not csv_paths:
            print("No CSV files found in the current directory.", file=sys.stderr)
            sys.exit(1)

    rerun = workflow_bundle = client_id = None
    if args.submit_to_comfyui:
        sys.path.insert(0, RERUN_SCRIPT_DIR)
        import rerun_prompts_comfyui as rerun

        workflow_path = Path(args.workflow).expanduser().resolve()
        workflow_bundle = rerun.load_workflow_bundle(workflow_path, args.server)
        client_id = str(uuid.uuid4())

    for csv_path in csv_paths:
        print(f"=== {csv_path} ===")
        process_csv(csv_path, args, rerun, workflow_bundle, client_id)


if __name__ == "__main__":
    main()
