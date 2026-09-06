"""
Reads Positive Prompt column from a ComfyUI prompt-log CSV, asks a local
Ollama model to rewrite each one as a natural-language prompt suitable for
a modern text-to-image model (e.g. Krea 2), and writes the result back into
a "Cleaned Prompt" column in the same CSV.

This script only ever cleans - it has no notion of content rating, and
never writes a "Content Rating"/"Rating Reason" column, nor does it submit
anything to ComfyUI. See rate_prompts.py for rating (rates "Cleaned
Prompt" when present, falling back to "Positive Prompt" otherwise) and
verify_prompt_ages.py for a second-pass age-requirement check, both run as
separate steps - or cleaning_orchestrator.py to run cleaning,
age-verification, rating, and (optionally) queuing each result to ComfyUI,
all in one command.

A row with no Positive Prompt text at all (extract_image_prompts.py writes
one of these for an image with no embedded metadata, instead of skipping
it - see that script's own docstring) falls back to describing the image
directly: a vision-capable Ollama model (VISION_MODEL) is shown the image
itself, via that row's "File Path" column, instead of having no text to
rewrite. This only ever triggers when there's truly no prompt text to work
with - a row that already has one always goes through the normal text
rewrite above, on --model or the default, unchanged.

A model occasionally leaks its base model's built-in safety refusal through
despite being an "abliterated"/uncensored fine-tune (observed verbatim from
gemma4-heretic: "I cannot fulfill this request. I am prohibited from
generating or rewriting content that contains sexually explicit material,
including descriptions of genitalia and pornographic acts."). This is
retried once immediately; if it persists, the row's Cleaned Prompt is set
to the literal string "ERROR" instead of being left blank. Unlike a normal
already-done row, "ERROR" is permanent and is never reprocessed by this
script again, --overwrite included, since a genuine refusal won't be fixed
by trying again later - rate_prompts.py and verify_prompt_ages.py both
recognize a Cleaned Prompt of "ERROR" the same way.

By default the final saved CSV is trimmed down to just the "Positive
Prompt" and "Cleaned Prompt" columns, dropping File Name/File Path/
Negative Prompt/etc. Pass -v/--verbose to keep every original column
instead. Either way, in-progress checkpoints during a run always keep
every column, so a crash mid-run never loses data for rows not yet
processed.

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

Also exposes a run(config_path) entry point (same JSON-config convention as
run_test.py/lora_test.py/generate_prompt_variations.py/rerun_prompts_
comfyui.py/extract_and_clean.py/cleaning_orchestrator.py) for driving this
programmatically without argparse.
"""

import argparse
import base64
import csv
import json
import os
import sys
import urllib.error
import urllib.request
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

try:
    from local_config import (  # run directly: python clean_prompts.py
        MAX_RESPONSE_TOKENS, REQUEST_TIMEOUT_SECONDS, is_refusal_response,
        load_local_text, load_text, sanitize_ascii, strip_thinking,
    )
except ImportError:
    from comfy_prompt_tools.local_config import (  # imported as a package
        MAX_RESPONSE_TOKENS, REQUEST_TIMEOUT_SECONDS, is_refusal_response,
        load_local_text, load_text, sanitize_ascii, strip_thinking,
    )

# clean_prompts.json (checked in) holds "system_prompt_base" - edit the
# prompt there, not in code. clean_prompts.local.json next to it (gitignored,
# same base+local pattern as everywhere else in this project) can add
# personal instructions on top via a "system_prompt_addendum" string. This is
# just the default - --prompt-config (CLI) / prompt_config (clean_all()/
# run()) points at a different <name>.json (+ optional <name>.local.json)
# instead, so a model that needs differently-worded directions (e.g. a
# reasoning model that responds better to an explicitly structured prompt)
# can have its own file without touching this one - see
# clean_prompts_qwen.json for an example.
SYSTEM_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "clean_prompts.json"
# A structured prompt config (see clean_prompts_qwen.json) ends its base
# text with this section header, right before where the model's answer is
# meant to begin - inserting an addendum there (as further response
# guidelines) keeps it a coherent part of the instructions instead of
# trailing after the model's told the task is already fully specified. An
# unstructured base (no such header - clean_prompts.json's plain-paragraph
# style) just gets the addendum appended at the end, as before.
_OUTPUT_SECTION_HEADER = "#OUTPUT:"

# A prompt config's base text can include this literal token wherever it
# wants the currently-selected visual style's instructions substituted in
# - see clean_prompts_qwen.json, which has this in place of what used to
# be a hardcoded "write only realistic photos" guideline. A config with no
# placeholder (e.g. the default clean_prompts.json) is simply unaffected -
# not every model config needs to support style-swapping. See
# style_realism.json/style_anime.json/style_oil_painting.json for the
# style side of this - --style-config (CLI) / style_config (clean_all()/
# run()) picks which one, defaulting to style_realism.json so a config
# using the placeholder keeps its original (realistic) behavior unless a
# different style is explicitly chosen.
STYLE_PLACEHOLDER = "{STYLE}"
DEFAULT_STYLE_CONFIG_PATH = Path(__file__).resolve().parent / "style_realism.json"


def resolve_style_adds(style_config=None):
    """style_config's "style_adds" text (default: DEFAULT_STYLE_CONFIG_PATH,
    i.e. realism), with <style_config>.local.json's own "style_adds" - if
    present - appended after it, same base+local pattern as a prompt
    config's addendum (just simple appending here, since a style's adds
    are a single self-contained instruction rather than a structured
    prompt with its own #OUTPUT: section to insert before)."""
    style_config = Path(style_config) if style_config else DEFAULT_STYLE_CONFIG_PATH
    base = load_text(style_config, "style_adds")
    local = load_local_text(style_config, "style_adds")
    return f"{base} {local}" if local else base


def build_system_prompt(config_path=None, style_config=None):
    """The base prompt (default: clean_prompts.json) is SFW - the age-safety
    clause in it stays baked in unconditionally, a guardrail rather than
    something droppable. If the base text contains STYLE_PLACEHOLDER, it's
    replaced first with resolve_style_adds(style_config) (default: realism)
    - done before the addendum step below since the placeholder lives
    inside the body, not at the end. <config_path>.local.json's addendum,
    if present, is then inserted just before a structured config's
    "#OUTPUT:" section (so it reads as more response guidelines, not a
    trailing afterthought) or appended as-is for an unstructured one."""
    config_path = Path(config_path) if config_path else SYSTEM_PROMPT_CONFIG_PATH
    base = load_text(config_path, "system_prompt_base")
    if STYLE_PLACEHOLDER in base:
        base = base.replace(STYLE_PLACEHOLDER, resolve_style_adds(style_config))
    addendum = load_local_text(config_path, "system_prompt_addendum")
    if not addendum:
        return base
    if _OUTPUT_SECTION_HEADER in base:
        before, after = base.split(_OUTPUT_SECTION_HEADER, 1)
        return f"{before.rstrip()}\n{addendum}\n\n{_OUTPUT_SECTION_HEADER}{after}"
    return base + " " + addendum


def resolve_system_prompts(config_path=None, style_config=None):
    """(system_prompt, image_description_system_prompt) for a given prompt
    config file (default: SYSTEM_PROMPT_CONFIG_PATH). clean_all()/run() call
    this once per invocation instead of relying on the module-level
    SYSTEM_PROMPT/IMAGE_DESCRIPTION_SYSTEM_PROMPT globals below, so
    --prompt-config actually takes effect - those globals stay only as the
    default for direct clean_prompt()/describe_image() calls (tests, a
    script run directly) that don't pass their own system_prompt.

    style_config: path to a style_<name>.json (default: realism) - only
    has any effect on a config_path whose base text contains
    STYLE_PLACEHOLDER; see build_system_prompt(). image_base can use the
    placeholder too, in principle, though none of the shipped configs do.

    This script has no notion of content rating at all - see rate_prompts.
    py for that, run separately (or via cleaning_orchestrator.py, which runs
    both plus verify_prompt_ages.py in sequence)."""
    config_path = Path(config_path) if config_path else SYSTEM_PROMPT_CONFIG_PATH
    base = build_system_prompt(config_path, style_config)
    image_base = load_text(config_path, "image_description_system_prompt")
    if STYLE_PLACEHOLDER in image_base:
        image_base = image_base.replace(STYLE_PLACEHOLDER, resolve_style_adds(style_config))
    return base, image_base


SYSTEM_PROMPT, IMAGE_DESCRIPTION_SYSTEM_PROMPT = resolve_system_prompts()

# sanitize_ascii moved to local_config.py - shared with rate_prompts.py,
# which needs it too (a rating "reason" is free-form model text just like a
# cleaned prompt, and crashes the same way printing to a Windows cp1252
# console if left unsanitized).



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





def _extract_cleaned_text(raw_response: str) -> str:
    """Strips a reasoning-model <think> block (see local_config.
    strip_thinking) and sanitizes to plain ASCII (see local_config.
    sanitize_ascii) - the only post-processing a plain rewrite/description
    response needs now that there's no rating to split out of it."""
    return sanitize_ascii(strip_thinking(raw_response).strip())


def _post_ollama(payload, timeout):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["response"]


def _call_with_refusal_retry(payload, timeout):
    """Posts to Ollama, retrying once if the raw response looks like the
    base model's safety filter refusing the request instead of actually
    attempting it (see local_config.is_refusal_response) - a resampled
    request often gets past it. Returns None if refusal persists after the
    retry, signalling the caller to mark the row ERROR instead of treating
    refusal text as if it were a real cleaned prompt/description."""
    raw_response = _post_ollama(payload, timeout)
    if is_refusal_response(raw_response):
        print("  refusal detected, retrying once...", file=sys.stderr)
        raw_response = _post_ollama(payload, timeout)
        if is_refusal_response(raw_response):
            return None
    return raw_response


def clean_prompt(positive_prompt: str, model: str = MODEL, system_prompt: str = None) -> str:
    """Returns the cleaned/rewritten text, or the literal string "ERROR" if
    the model's safety filter refused twice in a row (see
    _call_with_refusal_retry) - never a rating, this script has no notion
    of content rating at all any more (see rate_prompts.py). system_prompt
    defaults to the module-level SYSTEM_PROMPT (built from
    SYSTEM_PROMPT_CONFIG_PATH) - process_csv()/clean_all() pass their own,
    resolved from --prompt-config, instead of relying on that default."""
    payload = {
        "model": model,
        "prompt": positive_prompt,
        "system": system_prompt if system_prompt is not None else SYSTEM_PROMPT,
        "stream": False,
        # Some models (e.g. gemma4-heretic) support a hidden "thinking" pass
        # before the visible answer. Left on, they can burn the entire
        # num_predict budget on invisible reasoning and return a completely
        # empty response with done_reason "length" - same failure mode
        # generate_prompt_variations.py already disables this for.
        "think": False,
        "options": {"num_predict": MAX_RESPONSE_TOKENS},
    }
    raw_response = _call_with_refusal_retry(payload, REQUEST_TIMEOUT_SECONDS)
    if raw_response is None:
        return "ERROR"
    return _extract_cleaned_text(raw_response)


def describe_image(image_path, model: str = VISION_MODEL, system_prompt: str = None) -> str:
    """Ask a local vision-capable Ollama model to describe an image
    directly - the fallback for a row with no prompt text at all to
    rewrite (see the module docstring). Returns the description, same as
    clean_prompt() (including the "ERROR" sentinel on a persistent
    refusal). image_path is read and sent as base64, same request shape as
    clean_prompt() otherwise. A larger timeout than clean_prompt()'s, since
    a vision model's first pass over an image can take noticeably longer
    than a pure text rewrite. system_prompt defaults to the module-level
    IMAGE_DESCRIPTION_SYSTEM_PROMPT, same override convention as
    clean_prompt()."""
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "prompt": "Describe this image.",
        "system": system_prompt if system_prompt is not None else IMAGE_DESCRIPTION_SYSTEM_PROMPT,
        "images": [image_b64],
        "stream": False,
        "think": False,  # see clean_prompt()'s comment on this
        "options": {"num_predict": MAX_RESPONSE_TOKENS},
    }
    raw_response = _call_with_refusal_retry(payload, REQUEST_TIMEOUT_SECONDS + 60)
    if raw_response is None:
        return "ERROR"
    return _extract_cleaned_text(raw_response)


def process_csv(csv_path, args, system_prompt, image_system_prompt):
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
        existing_output = row.get(OUTPUT_COLUMN, "").strip()
        # ERROR is permanent (a refusal that survived its own dedicated
        # retry in _call_with_refusal_retry) - never retried, --overwrite
        # or not, same as rate_prompts.py treats its own ERROR rows.
        if existing_output == "ERROR":
            continue
        # A Cleaned Prompt that is itself refusal text (written by a run
        # from before this retry/ERROR logic existed) isn't a real "done"
        # row either - always worth a normal retry, same as ERROR would
        # have gotten if this logic had caught it the first time, but
        # without requiring --overwrite and redoing every real row too.
        if existing_output and not is_refusal_response(existing_output) and not args.overwrite:
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
                row[OUTPUT_COLUMN] = describe_image(image_path, system_prompt=image_system_prompt)
                row[PROMPT_COLUMN] = NOT_FOUND_MARKER
            else:
                row[OUTPUT_COLUMN] = clean_prompt(positive, model=args.model, system_prompt=system_prompt)
            succeeded += 1
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            row[OUTPUT_COLUMN] = ""
            failed += 1

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
            # the final saved file gets trimmed down to just the prompt columns.
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=[PROMPT_COLUMN, OUTPUT_COLUMN], extrasaction="ignore"
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


def clean_all(csv_paths, overwrite=False, verbose=False, model=MODEL, prompt_config=None, style_config=None):
    """Core logic behind main() - process an explicit list of CSV paths
    without going through argparse. Callable directly by other scripts
    (e.g. extract_and_clean.py, cleaning_orchestrator.py). prompt_config:
    path to a <name>.json prompt config (default: clean_prompts.json, i.e.
    SYSTEM_PROMPT_CONFIG_PATH) - see clean_prompts_qwen.json for an example
    of a second one, for a model that needs differently-worded directions.
    style_config: path to a style_<name>.json (default: style_realism.json)
    - only affects a prompt_config whose base text opts in via
    STYLE_PLACEHOLDER; see build_system_prompt(). Only ever produces a
    Cleaned Prompt - no rating, and nothing submitted to ComfyUI; see
    rate_prompts.py/verify_prompt_ages.py, or cleaning_orchestrator.py to
    run cleaning, age-verification, rating, and (optionally) queuing to
    ComfyUI all in sequence."""
    args = argparse.Namespace(overwrite=overwrite, verbose=verbose, model=model)
    system_prompt, image_system_prompt = resolve_system_prompts(prompt_config, style_config)

    for csv_path in csv_paths:
        print(f"=== {csv_path} ===")
        process_csv(csv_path, args, system_prompt, image_system_prompt)


def run(config_path):
    """JSON-config-driven entry point, same convention as run_test.py/
    lora_test.py/generate_prompt_variations.py/rerun_prompts_comfyui.py/
    extract_and_clean.py/cleaning_orchestrator.py - a caller like the GUI
    can drive this without going through argparse. Config file format:
        {
            "csv_paths": ["...", "..."],
            "model": "...",              // optional
            "prompt_config": "...",      // optional - path to a <name>.json prompt
                                          //   config (default: clean_prompts.json);
                                          //   see clean_prompts_qwen.json
            "style_config": "...",       // optional - path to a style_<name>.json
                                          //   (default: style_realism.json); only
                                          //   affects a prompt_config that opts in
                                          //   via {STYLE} in its base text
            "overwrite": false,          // optional
            "verbose": false              // optional
        }
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    clean_all(
        csv_paths=config["csv_paths"],
        overwrite=config.get("overwrite", False),
        verbose=config.get("verbose", False),
        model=config.get("model", MODEL),
        prompt_config=config.get("prompt_config"),
        style_config=config.get("style_config"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to the CSV file to process (default: every *.csv file in the current directory)")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows that already have a Cleaned Prompt value (default: skip them)")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model to use (default: {MODEL})")
    parser.add_argument("--prompt-config", default=None,
                         help="Path to a <name>.json prompt-directions config, base+local-overridable the same way "
                              f"as the default (default: {SYSTEM_PROMPT_CONFIG_PATH.name}) - use this to give a "
                              "different model (e.g. a reasoning model that needs more explicit structure) its own "
                              "wording without touching the default file. See clean_prompts_qwen.json for an example.")
    parser.add_argument("--style-config", default=None,
                         help=f"Path to a style_<name>.json (default: {DEFAULT_STYLE_CONFIG_PATH.name}) - only has "
                              "an effect on a --prompt-config whose base text opts in via a {STYLE} placeholder "
                              "(see clean_prompts_qwen.json). Swaps the requested visual style (realism, anime, "
                              "oil painting, ...) without editing the prompt config itself.")
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

    clean_all(
        csv_paths,
        overwrite=args.overwrite, verbose=args.verbose, model=args.model,
        prompt_config=args.prompt_config, style_config=args.style_config,
    )


if __name__ == "__main__":
    main()
