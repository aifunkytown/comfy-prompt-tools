# Example commands

A running log of real invocations used with this toolset, kept as a quick
reference for the typical workflow. Adjust paths/workflow files for your own
setup - see `README.md` for what each script does and its full `--help`.

## Extract image prompts

```bash
python "comfy_prompt_tools\extract_image_prompts.py"
python "comfy_prompt_tools\extract_image_prompts.py" "F:\Programs\ComfyFiles\output\2026-07-22"
python "comfy_prompt_tools\extract_image_prompts.py" "F:\Programs\ComfyFiles\output\SavedFromProfile"
```

## Extract + clean in one step

```bash
python "comfy_prompt_tools\extract_and_clean.py" "F:\Programs\ComfyFiles\output\2026-07-22"
python "comfy_prompt_tools\extract_and_clean.py" "F:\Programs\ComfyFiles\output\SavedFromProfile" --overwrite
```

(Errors out immediately if Ollama isn't reachable, before extracting
anything - same `--overwrite`/`-v`/`--submit-to-comfyui`/etc. options as
`clean_prompts.py` alone are accepted and passed through.)

## Rerun through ComfyUI

```bash
python "comfy_prompt_tools\rerun_prompts_comfyui.py" "F:\Programs\ComfyFiles\output\SavedFromProfile\SavedFromProfile-prompts.csv" 1 3 --workflow "F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json"

python "comfy_prompt_tools\rerun_prompts_comfyui.py" prompts.csv 1 3 --workflow "F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json"

python "comfy_prompt_tools\rerun_prompts_comfyui.py" "F:\Programs\ComfyFiles\output\Prompts\prompts.csv" 1 3 --workflow "F:\path\to\your_workflow_api.json"

python "comfy_prompt_tools\rerun_prompts_comfyui.py" "F:\Programs\ComfyFiles\output\2026-06-03-prompts.csv" --workflow "F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json"

python "comfy_prompt_tools\rerun_prompts_comfyui.py" "F:\Programs\ComfyFiles\output\SavedFromProfile\SavedFromProfile-prompts.csv.tmp" 99 220

python "comfy_prompt_tools\rerun_prompts_comfyui.py" "F:\Programs\ComfyFiles\output\general_solo.csv" 50 200
```

(`--workflow` defaults to `krea2_basic_t2i.json` and is auto-converted from
live ComfyUI on every run, so it can usually be omitted - keyword-triggered
LoRAs are handled automatically via `lora_rules.json`/`lora_rules.local.json`
in `rerun_prompts_comfyui.py`.)

## Cleanup (rewrite prompts via Ollama)

```bash
python "comfy_prompt_tools\clean_prompts.py" "F:\Programs\ComfyFiles\output\SavedFromProfile\SavedFromProfile-prompts.csv" --overwrite --submit-to-comfyui

python "comfy_prompt_tools\clean_prompts.py" "F:\Programs\ComfyFiles\output\2026-05-31-prompts.csv" --overwrite
```

## Variations

```bash
python "comfy_prompt_tools\generate_prompt_variations.py" "F:\Programs\ComfyFiles\output\SavedFromProfile\SavedFromProfile-prompts.csv" 156 "setting" 2
```
