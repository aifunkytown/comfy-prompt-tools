"""
Runs extract_image_prompts.py then cleaning_orchestrator.py in one step:
scan a directory for images, write/append one prompt CSV per folder, then
run the full clean -> age-verify -> content-rate pipeline (and, optionally,
queue to ComfyUI) on each CSV.

Ollama's reachability is checked up front, before anything is extracted -
if it isn't running, this exits with an error instead of doing the
extraction for nothing (the cleaning pipeline would just fail row-by-row
otherwise). Start it with `ollama serve` if you see that error (and make
sure the model is pulled: `ollama pull <model>`).

Usage:
    python extract_and_clean.py [directory] [-o output_dir]

    python extract_and_clean.py "F:\\Programs\\ComfyFiles\\output\\Prompts"

    # Run with no arguments to scan the current directory (and its
    # sub-directories), same default as extract_image_prompts.py alone.
    python extract_and_clean.py

    # Everything past the directory/-o is passed straight through to
    # clean_prompts.py's own processing - see its docstring for details:
    python extract_and_clean.py "F:\\Programs\\ComfyFiles\\output\\Prompts" --overwrite --verbose

    # Use a different local Ollama model instead of the default:
    python extract_and_clean.py "F:\\Programs\\ComfyFiles\\output\\Prompts" --model llama3.1:8b

    # Also submit each cleaned prompt to ComfyUI for rendering as it's
    # cleaned (keyword-matched LoRAs are turned on automatically, same as
    # rerun_prompts_comfyui.py):
    python extract_and_clean.py "F:\\Programs\\ComfyFiles\\output\\Prompts" \\
        --submit-to-comfyui --workflow <workflow.json>

Requires: everything extract_image_prompts.py and cleaning_orchestrator.py
each require - Pillow, and Ollama running locally with the model pulled.

Also exposes a run(config_path) entry point (same JSON-config convention as
run_test.py/lora_test.py/generate_prompt_variations.py/rerun_prompts_
comfyui.py/cleaning_orchestrator.py) for driving this programmatically
without argparse - used by funkytown-testing-harness-gui's Generations tab.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import clean_prompts
    import cleaning_orchestrator
    import extract_image_prompts
except ImportError:
    from comfy_prompt_tools import clean_prompts, cleaning_orchestrator, extract_image_prompts


def run_all(directory, output_dir=None, submit_to_comfyui=False, workflow=cleaning_orchestrator.DEFAULT_WORKFLOW,
            server=cleaning_orchestrator.DEFAULT_COMFYUI_SERVER, random_seed=False, overwrite=False, verbose=False,
            model=clean_prompts.MODEL, prompt_config=None, style_config=None):
    """Core logic behind main() and run() below - callable directly by other
    callers (e.g. the GUI) without going through argparse or a JSON file."""
    if not clean_prompts.check_ollama_running():
        sys.exit(
            f"Error: Ollama doesn't appear to be reachable at {clean_prompts.OLLAMA_URL} - "
            f"start it with `ollama serve` (and make sure the model is pulled: "
            f"`ollama pull {model}`), then try again."
        )

    csv_paths = extract_image_prompts.extract_all(directory, output_dir)
    if not csv_paths:
        return  # extract_all() already printed why (no images found)

    cleaning_orchestrator.run_all(
        [str(p) for p in csv_paths],
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


def run(config_path):
    """JSON-config-driven entry point, same convention as run_test.run()/
    lora_test.run()/generate_prompt_variations.run()/rerun_prompts_comfyui.run()
    - a caller like the GUI can drive this without going through argparse.
    Config file format:
        {
            "directory": "...",
            "output_dir": "...",        // optional
            "model": "...",              // optional
            "prompt_config": "...",      // optional - see cleaning_orchestrator.run()'s config docstring
            "style_config": "...",       // optional - see cleaning_orchestrator.run()'s config docstring
            "overwrite": false,          // optional
            "verbose": false,            // optional
            "submit_to_comfyui": false,  // optional
            "workflow": "...",           // optional, required if submit_to_comfyui
            "server": "...",             // optional
            "random_seed": false         // optional
        }
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_all(
        directory=config["directory"],
        output_dir=config.get("output_dir"),
        submit_to_comfyui=config.get("submit_to_comfyui", False),
        workflow=config.get("workflow", cleaning_orchestrator.DEFAULT_WORKFLOW),
        server=config.get("server", cleaning_orchestrator.DEFAULT_COMFYUI_SERVER),
        random_seed=config.get("random_seed", False),
        overwrite=config.get("overwrite", False),
        verbose=config.get("verbose", False),
        model=config.get("model", clean_prompts.MODEL),
        prompt_config=config.get("prompt_config"),
        style_config=config.get("style_config"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "directory", nargs="?", default=".",
        help="Directory to scan recursively for images (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory to write every extracted CSV file into (default: save each CSV into the folder it was "
             "scanned from). Created if it doesn't exist.",
    )
    parser.add_argument("--submit-to-comfyui", action="store_true",
                         help="After cleaning each row, submit the cleaned prompt to a running ComfyUI server via rerun_prompts_comfyui.py")
    parser.add_argument("--workflow", default=cleaning_orchestrator.DEFAULT_WORKFLOW,
                         help=f"Workflow JSON used for every row when --submit-to-comfyui is given (default: {cleaning_orchestrator.DEFAULT_WORKFLOW})")
    parser.add_argument("--server", default=cleaning_orchestrator.DEFAULT_COMFYUI_SERVER,
                         help=f"ComfyUI server URL for --submit-to-comfyui (default: {cleaning_orchestrator.DEFAULT_COMFYUI_SERVER})")
    parser.add_argument("--random-seed", action="store_true",
                         help="Randomize seed/noise_seed inputs when submitting to ComfyUI")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows that already have a Cleaned Prompt value (default: skip them)")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="Keep every original column in the saved CSV (default: trim the final output down to "
                              "just Positive Prompt and Cleaned Prompt)")
    parser.add_argument("--model", default=clean_prompts.MODEL,
                         help=f"Ollama model to use (default: {clean_prompts.MODEL})")
    parser.add_argument("--prompt-config", default=None,
                         help="Path to a <name>.json prompt-directions config - see clean_prompts.py's own "
                              "--prompt-config help for details")
    parser.add_argument("--style-config", default=None,
                         help="Path to a style_<name>.json - see clean_prompts.py's own --style-config help for details")
    args = parser.parse_args()

    run_all(
        directory=args.directory,
        output_dir=args.output_dir,
        submit_to_comfyui=args.submit_to_comfyui,
        workflow=args.workflow,
        server=args.server,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        verbose=args.verbose,
        model=args.model,
        prompt_config=args.prompt_config,
        style_config=args.style_config,
    )


if __name__ == "__main__":
    main()
