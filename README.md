# comfy-prompt-tools

A collection of standalone CLI scripts for extracting, cleaning, varying, and
resubmitting image generation prompts around a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
setup, plus a couple of helpers for pulling reference prompts from Civitai.

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

`playwright` is only needed for scripts that convert a saved (non-API-format)
ComfyUI workflow, or that drive a real browser against Civitai
(`civitai_login.py`, `scrape_civitai_prompts.py`).

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
  automatically (see `rerun_prompts_comfyui.py`'s `LORA_RULES`).

- **`generate_prompt_variations.py`** - Takes one row of a prompt CSV and asks
  Ollama to generate variations that change a specific described aspect (e.g.
  "dress color") while leaving the rest of the prompt alone. Supports a
  controlled vocabulary per aspect (`prompt_aspect_vocab.json`, colocated with
  this script), multi-value aspects, and randomly-chosen aspects.

### Submitting to ComfyUI

- **`rerun_prompts_comfyui.py`** - Resubmits every row of a prompt CSV (or
  every CSV in a directory) to a running ComfyUI server, swapping in each
  row's positive/negative prompt text against a workflow template that
  represents your current ComfyUI settings. Automatically turns on
  keyword-matched LoRAs (`LORA_RULES`) via the workflow's rgthree "Power Lora
  Loader" node. Accepts either a saved ComfyUI workflow (converted
  automatically via a headless browser) or an API-format export.

### CSV housekeeping

- **`dedupe_prompts_csv.py`** - Removes duplicate rows (by exact `Positive
  Prompt` match) within each CSV in the output folder, independently per file.
- **`combine_small_csvs.py`** - Merges CSVs with fewer than 30 rows together
  (pooled regardless of date adjacency) into combined files named after the
  newest file in each group.
- **`sort_prompts_by_category.py`** - Combines every CSV in the output folder
  into category files (`futa.csv` / `furry.csv` / `general_solo.csv` /
  `general_group.csv`) based on keywords and estimated subject count.

### Civitai reference scraping

- **`civitai_login.py`** - One-time interactive login helper: opens a real
  browser window for you to log in, then saves the session to
  `civitai_auth_state.json` (gitignored - **never commit this file**, it
  contains live auth cookies) for reuse by `scrape_civitai_prompts.py`.
- **`scrape_civitai_prompts.py`** - Scrapes prompts from a Civitai-style tag
  feed into a CSV, deduplicated by prompt text, using the saved login session.

Run any script with `--help` for its full argument list.

See [`EXAMPLES.md`](EXAMPLES.md) for real example invocations.
