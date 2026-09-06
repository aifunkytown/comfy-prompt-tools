"""
Runs clean_prompts.py, then verify_prompt_ages.py, then rate_prompts.py on
the same CSV(s) in one command: rewrite each Positive Prompt into a Cleaned
Prompt, make sure every subject in that cleaned text has an explicit age
stated (a safety net for clean_prompts.py's own age instructions, in case a
model doesn't follow them), then content-rate the final text.

Age-verification runs before rating, not after, so a row's Content Rating
reflects its actually-final Cleaned Prompt - this also matters for
correctness, not just tidiness: rate_prompts.json's REVIEW rubric has its
own exception for an explicitly stated adult age, which only works if that
age is already present in the text being rated.

None of the three scripts know about each other's job - clean_prompts.py
has no notion of content rating, and rate_prompts.py has no notion of
cleaning - this script is what chains them together. Each remains fully
usable on its own too (e.g. rate_prompts.py alone, to backfill ratings on
a CSV that predates this pipeline, or was hand-edited afterward).

Ollama's reachability is checked up front, before anything runs - if it
isn't running, this exits with an error instead of getting partway through
cleaning for nothing.

Usage:
    python clean_and_rate.py <path-to-csv>

    # Process every *.csv file in the current directory instead of one file:
    python clean_and_rate.py

    # Everything past the path is passed straight through to all three
    # scripts - see their own docstrings for details:
    python clean_and_rate.py <path-to-csv> --overwrite --verbose

    # Use a different local Ollama model instead of the default, for all
    # three stages:
    python clean_and_rate.py <path-to-csv> --model llama3.1:8b

    # Also submit each cleaned prompt to ComfyUI for rendering as it's
    # cleaned (keyword-matched LoRAs are turned on automatically, same as
    # rerun_prompts_comfyui.py) - this happens as part of the clean_prompts.py
    # stage, before age-verification/rating:
    python clean_and_rate.py <path-to-csv> --submit-to-comfyui --workflow <workflow.json>

Also exposes a run(config_path) entry point (same JSON-config convention as
run_test.py/lora_test.py/generate_prompt_variations.py/rerun_prompts_
comfyui.py/extract_and_clean.py) for driving this programmatically without
argparse.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import clean_prompts
    import rate_prompts
    import verify_prompt_ages
except ImportError:
    from comfy_prompt_tools import clean_prompts, rate_prompts, verify_prompt_ages


def run_all(csv_paths, submit_to_comfyui=False, workflow=clean_prompts.DEFAULT_WORKFLOW,
            server=clean_prompts.DEFAULT_COMFYUI_SERVER, random_seed=False, overwrite=False, verbose=False,
            model=clean_prompts.MODEL, prompt_config=None, style_config=None):
    """Core logic behind main() and run() below - callable directly by
    other callers (e.g. the GUI) without going through argparse or a JSON
    file. Uses the same model and overwrite setting for all three stages -
    call clean_prompts.clean_all()/verify_prompt_ages.verify_all()/
    rate_prompts.rate_all() directly instead if a stage needs its own
    model or independent overwrite behavior."""
    if not clean_prompts.check_ollama_running():
        sys.exit(
            f"Error: Ollama doesn't appear to be reachable at {clean_prompts.OLLAMA_URL} - "
            f"start it with `ollama serve` (and make sure the model is pulled: "
            f"`ollama pull {model}`), then try again."
        )

    csv_paths = [str(p) for p in csv_paths]

    clean_prompts.clean_all(
        csv_paths,
        submit_to_comfyui=submit_to_comfyui,
        workflow=workflow,
        server=server,
        random_seed=random_seed,
        overwrite=overwrite,
        verbose=verbose,
        model=model,
        prompt_config=prompt_config,
        style_config=style_config,
    )
    verify_prompt_ages.verify_all(csv_paths, model=model, overwrite=overwrite)
    rate_prompts.rate_all(csv_paths, model=model, overwrite=overwrite)


def run(config_path):
    """JSON-config-driven entry point, same convention as run_test.run()/
    lora_test.run()/generate_prompt_variations.run()/rerun_prompts_comfyui.
    run()/extract_and_clean.run() - a caller like the GUI can drive this
    without going through argparse. Config file format:
        {
            "csv_paths": ["...", "..."],
            "model": "...",              // optional - used for all three stages
            "prompt_config": "...",      // optional - see clean_prompts.run()'s config docstring
            "style_config": "...",       // optional - see clean_prompts.run()'s config docstring
            "overwrite": false,          // optional - used for all three stages
            "verbose": false,            // optional - clean_prompts.py stage only
            "submit_to_comfyui": false,  // optional - clean_prompts.py stage only
            "workflow": "...",           // optional, required if submit_to_comfyui
            "server": "...",             // optional
            "random_seed": false         // optional - clean_prompts.py stage only
        }
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_all(
        csv_paths=config["csv_paths"],
        submit_to_comfyui=config.get("submit_to_comfyui", False),
        workflow=config.get("workflow", clean_prompts.DEFAULT_WORKFLOW),
        server=config.get("server", clean_prompts.DEFAULT_COMFYUI_SERVER),
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
                         help="After cleaning each row, submit the cleaned prompt to a running ComfyUI server via rerun_prompts_comfyui.py")
    parser.add_argument("--workflow", default=clean_prompts.DEFAULT_WORKFLOW,
                         help=f"Workflow JSON used for every row when --submit-to-comfyui is given (default: {clean_prompts.DEFAULT_WORKFLOW})")
    parser.add_argument("--server", default=clean_prompts.DEFAULT_COMFYUI_SERVER,
                         help=f"ComfyUI server URL for --submit-to-comfyui (default: {clean_prompts.DEFAULT_COMFYUI_SERVER})")
    parser.add_argument("--random-seed", action="store_true",
                         help="Randomize seed/noise_seed inputs when submitting to ComfyUI")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows all three scripts would otherwise skip as already done (default: skip them)")
    parser.add_argument("--model", default=clean_prompts.MODEL,
                         help=f"Ollama model to use for all three stages (default: {clean_prompts.MODEL})")
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
