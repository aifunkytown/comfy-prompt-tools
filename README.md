# comfy-prompt-tools

A collection of standalone CLI scripts for extracting, cleaning, varying, and
resubmitting image generation prompts around a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
setup.

Each script is self-contained and runnable directly with `python`; nothing
here requires installing the package (though `pyproject.toml` is included if
you'd rather `pip install -e .`).

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally, with your chosen model pulled
  (`clean_prompts.py` and `generate_prompt_variations.py` default to
  `gemma4:12b` - override with `--model` or by editing the `MODEL` constant)
- A running ComfyUI server (for `rerun_prompts_comfyui.py` / `clean_prompts.py --submit-to-comfyui`)

```bash
pip install -r requirements.txt
playwright install chromium
```

`playwright` is only needed by `rerun_prompts_comfyui.py`, and only when
`--workflow` points at a saved (non-API-format) ComfyUI workflow rather than
an already-exported API-format file - see that script's `--help` for details.

### Local paths

Several scripts default to absolute paths under `F:\Programs\ComfyFiles\...`
(this author's ComfyUI install/output locations) and workflow filenames like
`krea2_basic_t2i.json`. Edit the constants near the top of each script (or
pass the equivalent CLI flag, where available) to point at your own setup.

## Scripts

### Prompt extraction

- **`extract_image_prompts.py`** - Scans a directory (recursively) for images
  and pulls the embedded generation prompt out of each one's metadata
  (ComfyUI, Automatic1111, or EXIF UserComment), writing one
  `<folder>-prompts.csv` per folder of images. Dedupes on exact `Positive
  Prompt` text within a folder; images with no positive prompt are skipped.

### Cleaning / rewriting

- **`clean_prompts.py`** - Reads a prompt CSV, asks a local Ollama model to
  rewrite each `Positive Prompt` into natural-language phrasing for a modern
  text-to-image model, and writes the result into a `Cleaned Prompt` column.
  Can optionally submit each cleaned prompt straight to ComfyUI as it's
  cleaned (`--submit-to-comfyui`), routing keyword-matched LoRAs on
  automatically (see `rerun_prompts_comfyui.py`'s `lora_rules.json`). Its
  own system prompt is likewise split base/local - `clean_prompts.local.json`
  next to it can append personal instructions.

- **`generate_prompt_variations.py`** - Takes one row of a prompt CSV and asks
  Ollama to generate variations that change a specific described aspect (e.g.
  "dress color") while leaving the rest of the prompt alone. Supports a
  controlled vocabulary per aspect (`prompt_aspect_vocab.json`, colocated with
  this script), multi-value aspects, and randomly-chosen aspects. Also
  exposes a `run(config_path)` entry point (same JSON-config convention as
  `funkytown-testing-harness`'s `run_test.py`/`lora_test.py`) for driving it
  programmatically - used by `funkytown-testing-harness-gui`'s Variations
  tab.

### Submitting to ComfyUI

- **`rerun_prompts_comfyui.py`** - Resubmits every row of a prompt CSV (or
  every CSV in a directory) to a running ComfyUI server, swapping in each
  row's positive/negative prompt text against a workflow template that
  represents your current ComfyUI settings. Automatically turns on
  keyword-matched LoRAs (`lora_rules.json`, empty by default - add a
  gitignored `lora_rules.local.json` for your own rules) via the workflow's
  rgthree "Power Lora Loader" node. Accepts either a saved ComfyUI workflow
  (converted automatically via a headless browser) or an API-format export.
  Also exposes a `run(config_path)` entry point (same JSON-config
  convention as `run_test.py`/`lora_test.py`/`generate_prompt_variations.py`)
  taking an explicit list of CSV paths rather than a single file-or-directory
  argument - used by `funkytown-testing-harness-gui`'s Variations tab to
  queue exactly the file(s) a just-finished run produced.

### CSV housekeeping

- **`dedupe_prompts_csv.py`** - Removes duplicate rows (by exact `Positive
  Prompt` match) within each CSV in the output folder, independently per file.
- **`combine_small_csvs.py`** - Merges CSVs with fewer than 30 rows together
  (pooled regardless of date adjacency) into combined files named after the
  newest file in each group.
- **`sort_prompts_by_category.py`** - Combines every CSV in the output folder
  into category files (`animal.csv` / `general_solo.csv` / `general_group.csv`
  by default) based on keywords and estimated subject count. The keyword
  lists live in `category_keywords.json` next to the script, not in the
  script itself - add a gitignored `category_keywords.local.json` next to it
  for personal categories/keyword overrides (same pattern as Claude Code's
  `settings.local.json`; `local_config.py` implements it and is reused by
  `rerun_prompts_comfyui.py`'s `lora_rules.json`/`lora_rules.local.json` and
  `clean_prompts.py`'s `clean_prompts.local.json`).

Run any script with `--help` for its full argument list.

See [`EXAMPLES.md`](EXAMPLES.md) for real example invocations.
