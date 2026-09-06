"""
Runs clean_prompts.py, then verify_prompt_ages.py, then rate_prompts.py -
and, optionally, then queues each result to ComfyUI - on the same CSV(s) in
one command: rewrite each Positive Prompt into a Cleaned Prompt, make sure
every subject in that cleaned text has an explicit age stated (a safety net
for clean_prompts.py's own age instructions, in case a model doesn't follow
them), content-rate the final text, and (if asked) submit it for rendering.

Age-verification runs before rating, not after, so a row's Content Rating
reflects its actually-final Cleaned Prompt - this also matters for
correctness, not just tidiness: rate_prompts.json's REVIEW rubric has its
own exception for an explicitly stated adult age, which only works if that
age is already present in the text being rated. Queuing to ComfyUI (if
requested) runs last of all, as its own separate pass over every CSV once
cleaning/verification/rating have all finished - so what actually gets
submitted for rendering is each row's fully-finished Cleaned Prompt, not an
in-progress draft from partway through the pipeline.

None of the four stages know about each other's job - clean_prompts.py has
no notion of content rating (or of ComfyUI at all), and rate_prompts.py has
no notion of cleaning - this script is what chains them together. Each
remains fully usable on its own too (e.g. rate_prompts.py alone, to
backfill ratings on a CSV that predates this pipeline, or was hand-edited
afterward).

Ollama's reachability is checked up front, before anything runs - if it
isn't running, this exits with an error instead of getting partway through
cleaning for nothing.

Usage:
    python cleaning_orchestrator.py <path-to-csv>

    # Process every *.csv file in the current directory instead of one file:
    python cleaning_orchestrator.py

    # Everything past the path is passed straight through to all three
    # Ollama-driven scripts - see their own docstrings for details:
    python cleaning_orchestrator.py <path-to-csv> --overwrite --verbose

    # Use a different local Ollama model instead of the default, for all
    # three Ollama-driven stages:
    python cleaning_orchestrator.py <path-to-csv> --model llama3.1:8b

    # Also submit each row's final Cleaned Prompt to ComfyUI for rendering,
    # once cleaning/age-verification/rating have all finished (keyword-matched
    # LoRAs are turned on automatically, same as rerun_prompts_comfyui.py):
    python cleaning_orchestrator.py <path-to-csv> --submit-to-comfyui --workflow <workflow.json>

Also exposes a run(config_path) entry point (same JSON-config convention as
run_test.py/lora_test.py/generate_prompt_variations.py/rerun_prompts_
comfyui.py/extract_and_clean.py) for driving this programmatically without
argparse.
"""

import argparse
import csv
import json
import sys
import urllib.error
import uuid
from pathlib import Path

try:
    import clean_prompts
    import rate_prompts
    import verify_prompt_ages
except ImportError:
    from comfy_prompt_tools import clean_prompts, rate_prompts, verify_prompt_ages

RERUN_SCRIPT_DIR = str(Path(__file__).resolve().parent)  # rerun_prompts_comfyui.py lives alongside this script
DEFAULT_WORKFLOW = r"F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json"
DEFAULT_COMFYUI_SERVER = "http://127.0.0.1:8000"


def submit_cleaned_prompt(rerun, workflow_bundle, server, client_id, row, cleaned_text, randomize_seed):
    """Queue one cleaned prompt on ComfyUI, reusing rerun_prompts_comfyui's
    workflow builder and its keyword-based LoRA toggling. Moved here from
    clean_prompts.py, which has no notion of ComfyUI at all any more - see
    queue_all()."""
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


def queue_all(csv_paths, workflow=DEFAULT_WORKFLOW, server=DEFAULT_COMFYUI_SERVER, random_seed=False):
    """Submits every row's current Cleaned Prompt to ComfyUI for rendering -
    a separate final pass over each CSV, run after clean_prompts.py/
    verify_prompt_ages.py/rate_prompts.py have all finished (see run_all()),
    so what actually gets queued is each row's fully-finished text rather
    than an in-progress draft. Skips a row with no Cleaned Prompt, or the
    permanent "ERROR" marker - nothing usable to submit."""
    sys.path.insert(0, RERUN_SCRIPT_DIR)
    import rerun_prompts_comfyui as rerun

    workflow_path = Path(workflow).expanduser().resolve()
    workflow_bundle = rerun.load_workflow_bundle(workflow_path, server)
    client_id = str(uuid.uuid4())

    for csv_path in csv_paths:
        print(f"=== Queuing {csv_path} ===")
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            cleaned_text = (row.get(clean_prompts.OUTPUT_COLUMN) or "").strip()
            if not cleaned_text or cleaned_text == "ERROR":
                continue
            submit_cleaned_prompt(rerun, workflow_bundle, server, client_id, row, cleaned_text, random_seed)


def run_all(csv_paths, submit_to_comfyui=False, workflow=DEFAULT_WORKFLOW, server=DEFAULT_COMFYUI_SERVER,
            random_seed=False, overwrite=False, verbose=False, model=clean_prompts.MODEL, prompt_config=None,
            style_config=None):
    """Core logic behind main() and run() below - callable directly by
    other callers (e.g. the GUI) without going through argparse or a JSON
    file. Uses the same model and overwrite setting for the clean/verify/
    rate stages - call clean_prompts.clean_all()/verify_prompt_ages.
    verify_all()/rate_prompts.rate_all() directly instead if a stage needs
    its own model or independent overwrite behavior."""
    if not clean_prompts.check_ollama_running():
        sys.exit(
            f"Error: Ollama doesn't appear to be reachable at {clean_prompts.OLLAMA_URL} - "
            f"start it with `ollama serve` (and make sure the model is pulled: "
            f"`ollama pull {model}`), then try again."
        )

    csv_paths = [str(p) for p in csv_paths]

    clean_prompts.clean_all(
        csv_paths,
        overwrite=overwrite,
        verbose=verbose,
        model=model,
        prompt_config=prompt_config,
        style_config=style_config,
    )
    verify_prompt_ages.verify_all(csv_paths, model=model, overwrite=overwrite)
    rate_prompts.rate_all(csv_paths, model=model, overwrite=overwrite)
    if submit_to_comfyui:
        queue_all(csv_paths, workflow=workflow, server=server, random_seed=random_seed)


def run(config_path):
    """JSON-config-driven entry point, same convention as run_test.run()/
    lora_test.run()/generate_prompt_variations.run()/rerun_prompts_comfyui.
    run()/extract_and_clean.run() - a caller like the GUI can drive this
    without going through argparse. Config file format:
        {
            "csv_paths": ["...", "..."],
            "model": "...",              // optional - used for the clean/verify/rate stages
            "prompt_config": "...",      // optional - see clean_prompts.run()'s config docstring
            "style_config": "...",       // optional - see clean_prompts.run()'s config docstring
            "overwrite": false,          // optional - used for the clean/verify/rate stages
            "verbose": false,            // optional - clean_prompts.py stage only
            "submit_to_comfyui": false,  // optional - runs queue_all() last, after clean/verify/rate
            "workflow": "...",           // optional, required if submit_to_comfyui
            "server": "...",             // optional
            "random_seed": false         // optional - queue_all() stage only
        }
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_all(
        csv_paths=config["csv_paths"],
        submit_to_comfyui=config.get("submit_to_comfyui", False),
        workflow=config.get("workflow", DEFAULT_WORKFLOW),
        server=config.get("server", DEFAULT_COMFYUI_SERVER),
        random_seed=config.get("random_seed", False),
        overwrite=config.get("overwrite", False),
        verbose=config.get("verbose", False),
        model=config.get("model", clean_prompts.MODEL),
        prompt_config=config.get("prompt_config"),
        style_config=config.get("style_config"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Path to the CSV file to process (default: every *.csv file in the current directory)")
    parser.add_argument("--submit-to-comfyui", action="store_true",
                         help="Once cleaning/age-verification/rating have all finished, submit each row's final "
                              "Cleaned Prompt to a running ComfyUI server via rerun_prompts_comfyui.py")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW,
                         help=f"Workflow JSON used for every row when --submit-to-comfyui is given (default: {DEFAULT_WORKFLOW})")
    parser.add_argument("--server", default=DEFAULT_COMFYUI_SERVER,
                         help=f"ComfyUI server URL for --submit-to-comfyui (default: {DEFAULT_COMFYUI_SERVER})")
    parser.add_argument("--random-seed", action="store_true",
                         help="Randomize seed/noise_seed inputs when submitting to ComfyUI")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows the clean/verify/rate stages would otherwise skip as already done "
                              "(default: skip them)")
    parser.add_argument("--model", default=clean_prompts.MODEL,
                         help=f"Ollama model to use for the clean/verify/rate stages (default: {clean_prompts.MODEL})")
    parser.add_argument("--prompt-config", default=None,
                         help="Path to a <name>.json prompt-directions config for the clean_prompts.py stage - see "
                              "clean_prompts.py's own --prompt-config help for details")
    parser.add_argument("--style-config", default=None,
                         help="Path to a style_<name>.json for the clean_prompts.py stage - see clean_prompts.py's "
                              "own --style-config help for details")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="Keep every original column in the clean_prompts.py stage's saved CSV (default: trim "
                              "down to just Positive Prompt + Cleaned Prompt before age-verification/rating add "
                              "their own columns)")
    args = parser.parse_args()

    if args.csv_path:
        csv_paths = [args.csv_path]
    else:
        csv_paths = sorted(str(p) for p in Path.cwd().glob("*.csv"))
        if not csv_paths:
            print("No CSV files found in the current directory.", file=sys.stderr)
            sys.exit(1)

    run_all(
        csv_paths,
        submit_to_comfyui=args.submit_to_comfyui, workflow=args.workflow, server=args.server,
        random_seed=args.random_seed, overwrite=args.overwrite, verbose=args.verbose, model=args.model,
        prompt_config=args.prompt_config, style_config=args.style_config,
    )


if __name__ == "__main__":
    main()
