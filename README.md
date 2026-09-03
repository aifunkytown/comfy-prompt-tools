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
  `gemma4:12b` - override with `--model` or by editing the `MODEL` constant).
  `clean_prompts.py` and `rerun_prompts_comfyui.py` also need
  `huihui_ai/qwen2.5-vl-abliterated:7b` pulled for their image-description
  fallback (a row with no prompt text at all - see each script below)
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
  Prompt` text within a folder; an image with no positive prompt metadata is
  still written, with that column empty and `File Path` pointing at the
  image - `clean_prompts.py` and `rerun_prompts_comfyui.py` both fall back to
  describing that image directly via a vision model instead of having
  nothing to work with (see each below).

### Cleaning / rewriting

- **`clean_prompts.py`** - Reads a prompt CSV, asks a local Ollama model to
  rewrite each `Positive Prompt` into natural-language phrasing for a modern
  text-to-image model, and writes the result into a `Cleaned Prompt` column.
  A row with no `Positive Prompt` text at all falls back to describing the
  image at that row's `File Path` directly, via a vision-capable model
  (`VISION_MODEL`, independent of `--model`) instead of being skipped - see
  `extract_image_prompts.py` above. That row's `Positive Prompt` is then set
  to the literal marker `not found` (rather than left blank) once described,
  so a later re-run recognizes it and repeats the *image* fallback instead
  of feeding that marker through the text-rewrite model as if it were a
  real prompt - the actual description lives in `Cleaned Prompt`, never
  cleaned twice. Can optionally submit each cleaned
  prompt straight to ComfyUI as it's cleaned (`--submit-to-comfyui`),
  routing keyword-matched LoRAs on automatically (see
  `rerun_prompts_comfyui.py`'s `lora_rules.json`). Its system prompts are
  likewise split base/local - both the text-rewrite prompt and the
  image-description prompt live in `clean_prompts.json` (checked in, edit
  them there rather than in code), and `clean_prompts.local.json` next to it
  can append personal instructions onto the text-rewrite one.

- **`extract_and_clean.py`** - Runs `extract_image_prompts.py` then
  `clean_prompts.py` in one command instead of two, on whatever CSV(s) the
  extraction step just wrote. Checks Ollama is actually reachable *before*
  extracting anything (errors out immediately if not, rather than
  extracting for nothing when cleaning would fail row-by-row anyway). Every
  `clean_prompts.py` option (`--overwrite`, `-v`/`--verbose`, `--model`,
  `--submit-to-comfyui`, `--workflow`, `--server`, `--random-seed`) is
  accepted and passed straight through.

- **`generate_prompt_variations.py`** - Takes one row of a prompt CSV and asks
  Ollama to generate variations that change a specific described aspect (e.g.
  "dress color") while leaving the rest of the prompt alone. Supports a
  controlled vocabulary per aspect (`prompt_aspect_vocab.json`, colocated with
  this script), multi-value aspects, and randomly-chosen aspects. Also
  exposes a `run(config_path)` entry point (same JSON-config convention as
  `funkytown-testing-harness`'s `run_test.py`/`lora_test.py`) for driving it
  programmatically - used by `funkytown-testing-harness-gui`'s Variations
  tab. One aspect, `"resolution"`, is special: rather than describing
  something to weave into the prompt text, each variation is just assigned
  one of the vocab's resolution values directly (never shown to the model),
  written to a `"Resolution"` column in the output CSV -
  `rerun_prompts_comfyui.py` (below) reads that column and resizes the
  workflow's Empty Latent Image node accordingly when re-queuing a row that
  has one set.

### Submitting to ComfyUI

- **`rerun_prompts_comfyui.py`** - Resubmits every row of a prompt CSV (or
  every CSV in a directory) to a running ComfyUI server, swapping in each
  row's positive/negative prompt text against a workflow template that
  represents your current ComfyUI settings. A row with no prompt text at all
  gets the same vision-model image-description fallback as `clean_prompts.py`
  (using that row's `File Path`) instead of being skipped - this works
  directly on a CSV straight out of `extract_image_prompts.py`, without ever
  going through `clean_prompts.py` first; the two scripts' fallbacks are
  independent of each other, so either still works entirely on its own. If
  the CSV *has* already been through `clean_prompts.py`, a row's `not found`
  `Positive Prompt` marker (see above) is never treated as real prompt text
  - the `Cleaned Prompt` column's actual description is used instead, with
  no redundant Ollama call. Automatically turns on
  keyword-matched LoRAs (`lora_rules.json`, empty by default - add a
  gitignored `lora_rules.local.json` for your own rules) via the workflow's
  rgthree "Power Lora Loader" node, and resizes the workflow's Empty Latent
  Image node per-row when a `"Resolution"` column is present (written by
  `generate_prompt_variations.py`'s `"resolution"` aspect - see above).
  Accepts either a saved ComfyUI workflow (converted automatically via a
  headless browser) or an API-format export. Also exposes a
  `run(config_path)` entry point (same JSON-config convention as
  `run_test.py`/`lora_test.py`/`generate_prompt_variations.py`) taking an
  explicit list of CSV paths rather than a single file-or-directory
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
