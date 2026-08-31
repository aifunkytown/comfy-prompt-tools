"""
Runs extract_image_prompts.py then clean_prompts.py in one step: scan a
directory for images, write/append one prompt CSV per folder, then ask a
local Ollama model to clean (rewrite) each CSV's Positive Prompt column
into a Cleaned Prompt column.

Ollama's reachability is checked up front, before anything is extracted -
if it isn't running, this exits with an error instead of doing the
extraction for nothing (cleaning would just fail row-by-row otherwise).
Start it with `ollama serve` if you see that error (and make sure the
model is pulled: `ollama pull <model>`).

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

Requires: everything extract_image_prompts.py and clean_prompts.py each
require - Pillow, and Ollama running locally with the model pulled.
"""

import argparse
import sys

try:
    import clean_prompts
    import extract_image_prompts
except ImportError:
    from comfy_prompt_tools import clean_prompts, extract_image_prompts


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
    parser.add_argument("--workflow", default=clean_prompts.DEFAULT_WORKFLOW,
                         help=f"Workflow JSON used for every row when --submit-to-comfyui is given (default: {clean_prompts.DEFAULT_WORKFLOW})")
    parser.add_argument("--server", default=clean_prompts.DEFAULT_COMFYUI_SERVER,
                         help=f"ComfyUI server URL for --submit-to-comfyui (default: {clean_prompts.DEFAULT_COMFYUI_SERVER})")
    parser.add_argument("--random-seed", action="store_true",
                         help="Randomize seed/noise_seed inputs when submitting to ComfyUI")
    parser.add_argument("--overwrite", action="store_true",
                         help="Reprocess rows that already have a Cleaned Prompt value (default: skip them)")
    parser.add_argument("-v", "--verbose", action="store_true",
                         help="Keep every original column in the saved CSV (default: trim the final output down to "
                              "just Positive Prompt and Cleaned Prompt)")
    parser.add_argument("--model", default=clean_prompts.MODEL,
                         help=f"Ollama model to use (default: {clean_prompts.MODEL})")
    args = parser.parse_args()

    if not clean_prompts.check_ollama_running():
        sys.exit(
            f"Error: Ollama doesn't appear to be reachable at {clean_prompts.OLLAMA_URL} - "
            f"start it with `ollama serve` (and make sure the model is pulled: "
            f"`ollama pull {args.model}`), then try again."
        )

    csv_paths = extract_image_prompts.extract_all(args.directory, args.output_dir)
    if not csv_paths:
        return  # extract_all() already printed why (no images found)

    clean_prompts.clean_all(
        [str(p) for p in csv_paths],
        submit_to_comfyui=args.submit_to_comfyui,
        workflow=args.workflow,
        server=args.server,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
        verbose=args.verbose,
        model=args.model,
    )


if __name__ == "__main__":
    main()
