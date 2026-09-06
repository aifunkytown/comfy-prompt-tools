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
  (ComfyUI, Automatic1111, SwarmUI, InvokeAI, or EXIF UserComment), writing
  one `<folder>-prompts.csv` per folder of images. SwarmUI writes its
  metadata into the same PNG "parameters" chunk A1111 uses, but as JSON -
  detected by content (a "sui_image_params" key), not assumed from the
  chunk name, so it's tried first and only falls through to plain A1111
  parsing when that JSON shape isn't present. Dedupes on exact `Positive
  Prompt` text within a folder; an image with no positive prompt metadata is
  still written, with that column empty and `File Path` pointing at the
  image - `clean_prompts.py` and `rerun_prompts_comfyui.py` both fall back to
  describing that image directly via a vision model instead of having
  nothing to work with (see each below).

- **`fix_swarmui_prompts.py`** - One-time repair for CSVs extracted *before*
  `extract_image_prompts.py` learned to recognize SwarmUI's format - those
  rows have the raw, unparsed SwarmUI JSON blob sitting in `Positive Prompt`
  instead of the actual prompt text. Finds rows in that shape and replaces
  `Positive Prompt`/`Negative Prompt`/`Other Parameters`/`Source Format`
  with the correctly extracted values (recomputing the prompt hash too);
  any stale `Cleaned Prompt` generated from the broken input is cleared so
  `clean_prompts.py` picks the row back up automatically next run, no
  `--overwrite` needed. Rows not in that shape are left completely
  untouched. `--move-to <path>` instead pulls fixed rows out of their
  source CSV entirely and collects them into one destination CSV, for
  isolating exactly the rows that need re-cleaning without disturbing the
  rest of a source file.

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
  cleaned twice. Only ever cleans - no notion of content rating, and
  doesn't submit anything to ComfyUI either; see `rate_prompts.py`/
  `verify_prompt_ages.py` below and `cleaning_orchestrator.py` to run
  cleaning, age-verification, rating, and (optionally) queuing to ComfyUI
  all together. Its system prompts are split base/local - both the
  text-rewrite prompt and the image-description prompt live in
  `clean_prompts.json` (checked in, edit them there rather than in code),
  and `clean_prompts.local.json` next to it can append personal
  instructions onto the text-rewrite one. Also exposes a `run(config_path)`
  entry point (same JSON-config convention as `run_test.py`/`lora_test.py`/
  `generate_prompt_variations.py`/`rerun_prompts_comfyui.py`/
  `extract_and_clean.py`/`cleaning_orchestrator.py`).

- **`rate_prompts.py`** - Content-rates a prompt CSV on a movie-style scale
  (`G`/`PG`/`PG-13`/`R`/`X`/`XXX`, or `REVIEW` for suspected underage
  content) via a local Ollama model, writing `Content Rating` and `Rating
  Reason` columns - rates `Cleaned Prompt` when present, falling back to
  `Positive Prompt` otherwise (matching what actually gets sent to
  ComfyUI). Has no notion of cleaning at all - run it directly to rate any
  CSV with prompt text, cleaned or not, or use `cleaning_orchestrator.py`
  to clean and rate together. Same base/local split as everywhere else:
  the rubric lives in `rate_prompts.json` (checked in),
  `rate_prompts.local.json` next to it can append personal instructions.

- **`verify_prompt_ages.py`** - Second-pass safety net for `clean_prompts.py`'s
  age requirement (every person/humanoid subject must have an explicit
  numeric age of at least 20 stated, 35 if mature/older, regardless of
  whether the image is explicit - see `clean_prompts.json`). Re-sends each
  row's `Cleaned Prompt` to a local Ollama model to check that requirement
  is actually met; if it is, the text comes back unchanged, otherwise just
  enough is rewritten to add/correct the age(s), leaving everything else
  untouched. A conservative local pre-filter skips the LLM call entirely
  when the scene is confidently single-subject and already has a clear
  age stated. Marks `Age Verified` = `yes` per row (skipped on a re-run
  unless `--overwrite`) plus a short `Age Verification Note`. Exists to
  backfill CSVs cleaned before this policy existed, or catch a model not
  following its own system prompt - not a replacement for
  `clean_prompts.py`'s own age instructions, which do the real work on a
  normal cleaning pass. Same refusal-retry/permanent-`ERROR` handling as
  `clean_prompts.py`/`rate_prompts.py`; same base/local system-prompt split
  (`verify_prompt_ages.json` + `verify_prompt_ages.local.json`).

- **`cleaning_orchestrator.py`** - Runs `clean_prompts.py`, then
  `verify_prompt_ages.py`, then `rate_prompts.py`, then (if
  `--submit-to-comfyui`) queues each row's final `Cleaned Prompt` to
  ComfyUI, in one command, on the same CSV(s). Age-verification runs
  before rating (not after) so a row's `Content Rating` reflects its
  actually-final `Cleaned Prompt` - this also matters for correctness:
  `rate_prompts.json`'s `REVIEW` exception for an explicitly stated adult
  age only works if that age is already present in the text being rated.
  Queuing to ComfyUI runs last of all, as its own separate pass once
  cleaning/verification/rating have all finished, reusing
  `rerun_prompts_comfyui.py`'s workflow builder and keyword-based LoRA
  routing - so what's actually submitted for rendering is each row's
  fully-finished text, not an in-progress draft. None of the four stages
  know about each other's job - this is what chains them together; each
  remains fully usable on its own too. Uses the same model and
  `--overwrite` for the clean/verify/rate stages - call the individual
  scripts directly instead if a stage needs its own model or independent
  overwrite behavior. Also exposes a `run(config_path)` entry point (same
  JSON-config convention as everywhere else).

- **`extract_and_clean.py`** - Runs `extract_image_prompts.py` then
  `cleaning_orchestrator.py` (clean -> verify -> rate -> optionally queue)
  in one command instead of two, on whatever CSV(s) the extraction step
  just wrote. Checks Ollama is actually reachable *before* extracting
  anything (errors out immediately if not, rather than extracting for
  nothing when the cleaning pipeline would fail row-by-row anyway). Every
  `cleaning_orchestrator.py` option (`--overwrite`, `-v`/`--verbose`,
  `--model`, `--submit-to-comfyui`, `--workflow`, `--server`,
  `--random-seed`) is accepted and passed straight through. Also exposes a
  `run(config_path)` entry point (same JSON-config convention as
  `run_test.py`/`lora_test.py`/`generate_prompt_variations.py`/
  `rerun_prompts_comfyui.py`) - used by
  `funkytown-testing-harness-gui`'s Generations tab.

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
- **`sort_prompts_by_category.py`** - Combines every CSV in a given directory
  (current directory by default; pass a directory and optionally
  `--output-dir` to use different source/output locations) into category
  files (`animal.csv` / `general_solo.csv` / `general_group.csv` by default)
  based on keywords and estimated subject count. Re-running it merges newly
  sorted rows into each category file's existing content rather than
  overwriting it. The keyword lists live in `category_keywords.json` next to
  the script, not in the script itself - add a gitignored
  `category_keywords.local.json` next to it for personal categories/keyword
  overrides (same pattern as Claude Code's `settings.local.json`;
  `local_config.py` implements it and is reused by `rerun_prompts_comfyui.py`'s
  `lora_rules.json`/`lora_rules.local.json` and `clean_prompts.py`'s
  `clean_prompts.local.json`).

Run any script with `--help` for its full argument list.

See [`EXAMPLES.md`](EXAMPLES.md) for real example invocations.
