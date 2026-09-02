"""
Read a CSV produced by extract_image_prompts.py and resubmit each prompt to a
running ComfyUI server, using a workflow template that represents your
*current* ComfyUI settings (checkpoint, sampler, steps, etc.) - only the
positive/negative prompt text is swapped in per row.

csv_file may also be a directory. In that case every *.csv file found
directly inside it (not counting rerun_log.csv) is processed, one after
another.

The workflow is loaded and converted to API format up front, before anything
is queued (see "Getting the workflow template" below for why: browser-based
conversion of a saved workflow isn't safe to interleave with an
already-progressing queue).

LoRA routing by prompt content
-------------------------------
A single workflow is used for every row (--workflow). Each row's prompt is
checked (case-insensitive) for certain keywords and, when matched, the
corresponding LoRA is turned on (via the workflow's rgthree "Power Lora
Loader" node) at a preset strength - see lora_rules.json (next to this
script) for the rule list, which ships empty in the public repo; a
gitignored lora_rules.local.json next to it adds your own personal rules the
same way category_keywords.local.json does for sort_prompts_by_category.py.
Multiple rules can match the same prompt at once, in which case every
matching LoRA is turned on together. Nothing else about the workflow
changes.

Image-description fallback for rows with no prompt metadata
-------------------------------------------------------------
A row with no Positive Prompt text at all - extract_image_prompts.py
writes one of these for an image with no embedded metadata, rather than
skipping it (see that script's own docstring) - is described directly
instead, via a vision-capable Ollama model (clean_prompts.VISION_MODEL)
shown the image at that row's "File Path" column. The generated
description is then used as that row's prompt text exactly like any other
row's. A row that already has prompt text is never affected by this - it's
purely a fallback for rows with nothing else to go on, and this script
still works standalone on a CSV that never goes through clean_prompts.py
at all.

Resolution from generate_prompt_variations.py's "resolution" aspect
------------------------------------------------------------------
If a CSV row has a non-empty "Resolution" column (written by
generate_prompt_variations.py when its special "resolution" aspect was
used - see that script's own docstring), its "WIDTHxHEIGHT" value is
applied to the workflow's Empty Latent Image node (or a same-shaped
variant, e.g. EmptySD3LatentImage) before queuing that row - everything
else about the workflow (batch_size included) is left as-is. A row with no
Resolution value (column absent, or blank on an unvaried "original" row)
leaves the workflow's own size alone.

Getting the workflow template
------------------------------
--workflow can point at either:
  - A regular saved ComfyUI workflow (Workflow menu -> Save, or one of the
    files under ComfyUI's user/default/workflows folder). The script will
    automatically convert it using ComfyUI's own conversion logic, run via
    a headless browser pointed at --server, so bypassed nodes, primitive
    nodes, and subgraphs are all resolved correctly. Requires:
        pip install playwright
        playwright install chromium
  - An already API-format export (Workflow menu -> Export (API), only
    visible once "Dev mode Options" is enabled in ComfyUI's settings). Used
    as-is, no browser/conversion needed.

This is fire-and-forget: each prompt is queued on the ComfyUI server and the
script moves on immediately without waiting for the image to render.
ComfyUI processes its queue on its own; check the ComfyUI window or output
folder for results.

Usage
-----
    python rerun_prompts_comfyui.py prompts.csv --workflow my_workflow_api.json

    python rerun_prompts_comfyui.py prompts.csv --workflow my_workflow_api.json \\
        --server http://127.0.0.1:8000 --random-seed --delay 1.0

    # Only rerun rows 3 through 7 of the CSV (1-indexed, inclusive):
    python rerun_prompts_comfyui.py prompts.csv 3 7 --workflow my_workflow_api.json

    # Directory given: run every *.csv directly inside it
    python rerun_prompts_comfyui.py "F:\\Programs\\ComfyFiles\\output\\SavedFromProfile"

Requires: only the Python standard library, unless --workflow points at a
saved (non-API-format) workflow, in which case playwright is also needed
(see above).
"""

import argparse
import copy
import csv
import datetime
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

try:
    from local_config import load_named_list  # run directly: python rerun_prompts_comfyui.py
    from clean_prompts import describe_image, VISION_MODEL
except ImportError:
    from comfy_prompt_tools.local_config import load_named_list  # imported as a package
    from comfy_prompt_tools.clean_prompts import describe_image, VISION_MODEL

LORA_RULES_PATH = Path(__file__).resolve().parent / "lora_rules.json"


def find_prompt_node_ids(workflow):
    """Return (positive_node_id, negative_node_id) within an API-format workflow."""
    for node in workflow.values():
        if "KSampler" in node.get("class_type", ""):
            inputs = node.get("inputs", {})
            pos = inputs.get("positive")
            neg = inputs.get("negative")
            pos_id = str(pos[0]) if isinstance(pos, list) and pos else None
            neg_id = str(neg[0]) if isinstance(neg, list) and neg else None
            if pos_id or neg_id:
                return pos_id, neg_id

    # Fallback: no KSampler found, just grab CLIPTextEncode nodes in order
    ids = [node_id for node_id, node in workflow.items() if node.get("class_type") == "CLIPTextEncode"]
    positive = ids[0] if len(ids) >= 1 else None
    negative = ids[1] if len(ids) >= 2 else None
    return positive, negative


def find_save_image_node_ids(workflow):
    return [
        node_id for node_id, node in workflow.items()
        if "SaveImage" in node.get("class_type", "")
    ]


def find_power_lora_loader_id(workflow):
    """Find the rgthree Power Lora Loader node by structure - each LoRA slot is an
    input whose value is a dict with 'on'/'lora'/'strength' keys - rather than by
    an exact class_type string, which isn't confirmed and may vary by version."""
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        if any(
            isinstance(v, dict) and {"on", "lora", "strength"} <= set(v.keys())
            for v in inputs.values()
        ):
            return node_id
    return None


def find_empty_latent_image_node(workflow):
    """Find the Empty Latent Image node (or a same-shaped variant, e.g.
    EmptySD3LatentImage) - class_type name first, falling back to structure
    (integer width/height/batch_size inputs) since the exact class name
    varies by model family/custom node pack, same approach as
    find_power_lora_loader_id."""
    for node_id, node in workflow.items():
        if "EmptyLatentImage" in node.get("class_type", ""):
            return node_id
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        if all(isinstance(inputs.get(k), (int, float)) for k in ("width", "height", "batch_size")):
            return node_id
    return None


def find_seed_inputs(workflow):
    """Return list of (node_id, input_key) for widget inputs that look like a seed."""
    seed_inputs = []
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for key in ("seed", "noise_seed"):
            if key in inputs and isinstance(inputs[key], (int, float)):
                seed_inputs.append((node_id, key))
    return seed_inputs


def apply_loras(wf, lora_node_id, lora_matches):
    """Turn on (and set the strength of) each matched LoRA slot within the Power
    Lora Loader node. Slots are matched by their 'lora' filename value rather than
    a positional/key naming scheme (e.g. 'lora_1'), since that isn't confirmed for
    this node type and shouldn't be guessed at."""
    if not lora_node_id or not lora_matches:
        return
    inputs = wf[lora_node_id].get("inputs", {})
    for lora_filename, strength in lora_matches:
        for slot in inputs.values():
            if isinstance(slot, dict) and slot.get("lora") == lora_filename:
                slot["on"] = True
                slot["strength"] = strength
                break
        else:
            print(f"  warning: LoRA slot for '{lora_filename}' not found in workflow", file=sys.stderr)


def set_resolution(wf, latent_node_id, width, height):
    """Set the Empty Latent Image node's width/height, leaving batch_size
    and everything else about the workflow untouched."""
    if not latent_node_id:
        print("  warning: Resolution given but no Empty Latent Image node found in workflow", file=sys.stderr)
        return
    inputs = wf[latent_node_id]["inputs"]
    inputs["width"] = width
    inputs["height"] = height


def build_workflow_for_row(template, positive_id, negative_id, save_ids, positive_text, negative_text, prefix, randomize_seed, lora_node_id=None, lora_matches=None, latent_node_id=None, resolution=None):
    wf = copy.deepcopy(template)

    if positive_id and positive_text:
        wf[positive_id]["inputs"]["text"] = positive_text
    if negative_id and negative_text:
        wf[negative_id]["inputs"]["text"] = negative_text

    for save_id in save_ids:
        wf[save_id]["inputs"]["filename_prefix"] = prefix

    if randomize_seed:
        for node_id, key in find_seed_inputs(wf):
            wf[node_id]["inputs"][key] = random.randint(0, 2**32 - 1)

    apply_loras(wf, lora_node_id, lora_matches)

    if resolution:
        try:
            width_str, height_str = resolution.lower().split("x")
            set_resolution(wf, latent_node_id, int(width_str), int(height_str))
        except ValueError:
            print(f"  warning: invalid Resolution value '{resolution}' - expected WIDTHxHEIGHT, e.g. '1920x1080'", file=sys.stderr)

    return wf


def is_api_format(data):
    """True if data looks like a flat API-format workflow (node-id -> {class_type, inputs})."""
    return isinstance(data, dict) and bool(data) and all(
        isinstance(v, dict) and "class_type" in v for v in data.values()
    )


def convert_ui_workflow_via_browser(server, ui_workflow):
    """Convert a saved (UI-format) ComfyUI workflow into API-format by driving
    ComfyUI's own frontend conversion logic (app.loadGraphData / app.graphToPrompt)
    through a headless browser. This correctly handles bypassed nodes, primitive
    nodes, and subgraphs, since it's the exact same code ComfyUI's UI uses."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(server, wait_until="domcontentloaded")
            page.wait_for_function(
                "window.app && typeof window.app.graphToPrompt === 'function'", timeout=30000
            )
            return page.evaluate(
                """async (wf) => {
                    await window.app.loadGraphData(wf);
                    const result = await window.app.graphToPrompt();
                    return result.output;
                }""",
                ui_workflow,
            )
        finally:
            browser.close()


def load_workflow_template(workflow_path, server):
    raw = json.loads(workflow_path.read_text(encoding="utf-8"))

    if "prompt" in raw and isinstance(raw["prompt"], dict) and is_api_format(raw["prompt"]):
        return raw["prompt"]  # unwrap if the whole /prompt payload was exported

    if is_api_format(raw):
        return raw

    print(f"{workflow_path} looks like a saved ComfyUI workflow (not API format).")
    print(f"Converting it via ComfyUI at {server} ...")
    try:
        template = convert_ui_workflow_via_browser(server, raw)
    except ImportError:
        print(
            "Error: converting a saved workflow requires playwright.\n"
            "    pip install playwright\n"
            "    playwright install chromium\n"
            "Alternatively, export the workflow in API format yourself: in ComfyUI, enable\n"
            "'Dev mode Options' in Settings, then use 'Export (API)' in the Workflow menu.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error: failed to convert workflow via browser: {e}", file=sys.stderr)
        sys.exit(1)

    if not is_api_format(template):
        print("Error: conversion did not produce a valid API-format workflow.", file=sys.stderr)
        sys.exit(1)

    print("Conversion succeeded.")
    return template


# Keyword -> LoRA mapping - see lora_rules.json (+ your own
# lora_rules.local.json) next to this script, not hardcoded here. Every rule
# whose keyword(s) appear (case-insensitive) in a prompt gets its LoRA
# turned on; multiple matching rules stack together on the same image.
LORA_RULES = load_named_list(LORA_RULES_PATH, "rules", "lora")


def select_loras(prompt_text):
    """Return [(lora_filename, strength), ...] for every LORA_RULES entry whose
    keyword(s) appear in prompt_text."""
    lower = prompt_text.lower()
    return [
        (rule["lora"], rule["strength"])
        for rule in LORA_RULES
        if any(k in lower for k in rule["keywords"])
    ]


def load_workflow_bundle(workflow_path, server):
    """Load a workflow file (converting it if needed) and return its template
    plus positive/negative/SaveImage/Power-Lora-Loader/Empty-Latent-Image
    node ids."""
    if not workflow_path.is_file():
        print(f"Error: workflow file not found: {workflow_path}", file=sys.stderr)
        sys.exit(1)

    template = load_workflow_template(workflow_path, server)
    positive_id, negative_id = find_prompt_node_ids(template)
    if not positive_id:
        print(f"Error: could not find a CLIPTextEncode/positive prompt node in {workflow_path}", file=sys.stderr)
        sys.exit(1)
    save_ids = find_save_image_node_ids(template)
    lora_node_id = find_power_lora_loader_id(template)
    latent_node_id = find_empty_latent_image_node(template)

    print(f"[{workflow_path.name}] positive node: {positive_id}   negative node: {negative_id or '(none found)'}")
    print(f"[{workflow_path.name}] SaveImage node(s): {save_ids or '(none found - filename_prefix will not be set)'}")
    print(f"[{workflow_path.name}] Power Lora Loader node: {lora_node_id or '(none found - keyword LoRAs will not be applied)'}")
    print(f"[{workflow_path.name}] Empty Latent Image node: {latent_node_id or '(none found - Resolution column values will not be applied)'}")

    return template, positive_id, negative_id, save_ids, lora_node_id, latent_node_id


def queue_prompt(server, workflow, client_id):
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{server}/prompt", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_csv_files(csv_dir):
    """All *.csv files directly inside csv_dir, excluding this script's own log output."""
    return sorted(
        p for p in csv_dir.glob("*.csv")
        if p.is_file() and p.name.lower() != "rerun_log.csv"
    )


def load_csv_rows(csv_path, start_row, end_row):
    """Return (cleaned_col, [(row_num, row_dict), ...]) for csv_path, honoring an
    optional 1-indexed inclusive row range."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    cleaned_col = next((c for c in fieldnames if "cleaned" in c.lower()), None)

    if start_row is not None or end_row is not None:
        first = start_row if start_row is not None else 1
        last = end_row if end_row is not None else len(rows)
        if first < 1 or last < first:
            print(f"Error: invalid row range {first}-{last} ({csv_path.name} has {len(rows)} data row(s))", file=sys.stderr)
            sys.exit(1)
        rows = rows[first - 1: last]
    else:
        first = 1

    numbered_rows = list(enumerate(rows, start=first))
    return cleaned_col, numbered_rows


def extract_prompt_text(row, cleaned_col):
    """Return (positive_text, negative_text, notes, resolution, image_path)
    for a CSV row - resolution is generate_prompt_variations.py's optional
    "Resolution" column ("WIDTHxHEIGHT", or "" if that aspect wasn't used/
    this is an unvaried "original" row). image_path is the row's "File
    Path", used as a describe_image() fallback (see run_batch) when
    positive_text is empty - extract_image_prompts.py writes a row like
    that for an image with no embedded prompt metadata, instead of
    skipping it."""
    cleaned_text = (row.get(cleaned_col) or "").strip() if cleaned_col else ""
    positive_text = cleaned_text or (row.get("Positive Prompt") or "").strip()
    negative_text = (row.get("Negative Prompt") or "").strip()
    notes = (row.get("Notes") or "").strip()
    resolution = (row.get("Resolution") or "").strip()
    image_path = (row.get("File Path") or "").strip()
    return positive_text, negative_text, notes, resolution, image_path


def run_batch(csv_paths, workflow_path, server="http://127.0.0.1:8000", start_row=None, end_row=None, random_seed=False, delay=0.2, log_path=None):
    """Core logic shared by the CLI (main()) and run(config_path) below.
    csv_paths is an explicit list of Path objects (a directory should already
    be expanded via discover_csv_files by the caller) - start_row/end_row
    only apply when there's exactly one path (ambiguous otherwise), same as
    the CLI's own restriction."""
    if (start_row is not None or end_row is not None) and len(csv_paths) != 1:
        print("Error: start_row/end_row require exactly one CSV file, not several (ambiguous across multiple CSVs)", file=sys.stderr)
        sys.exit(1)
    if not csv_paths:
        print("Error: no CSV file(s) given", file=sys.stderr)
        sys.exit(1)

    # Phase 1: read every row from every CSV up front (no network activity yet).
    entries = []  # (csv_path, row_num, end_row_num, name, positive_text, negative_text, notes, resolution, image_path)
    for csv_path in csv_paths:
        cleaned_col, numbered_rows = load_csv_rows(csv_path, start_row, end_row)
        if cleaned_col:
            print(f"[{csv_path.name}] Using '{cleaned_col}' column in place of 'Positive Prompt' where present")
        if not numbered_rows:
            continue
        end_row_num = numbered_rows[-1][0]
        for row_num, row in numbered_rows:
            name = row.get("File Name", f"row{row_num}")
            positive_text, negative_text, notes, resolution, image_path = extract_prompt_text(row, cleaned_col)
            entries.append((csv_path, row_num, end_row_num, name, positive_text, negative_text, notes, resolution, image_path))

    # Phase 2: load/convert the workflow once, before queuing a single prompt. Once
    # prompts start landing in ComfyUI's queue, driving the browser to convert a
    # saved workflow is no longer safe to interleave with that.
    template, positive_id, negative_id, save_ids, lora_node_id, latent_node_id = load_workflow_bundle(workflow_path, server)

    # Phase 3: queue everything. Purely stdlib HTTP calls from here on - no browser.
    log_path = Path(log_path) if log_path else csv_paths[0].parent / "rerun_log.csv"
    client_id = str(uuid.uuid4())

    print(f"\nQueuing {len(entries)} row(s) total...")
    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["CSV File", "File Name", "Status", "Prompt ID", "LoRAs", "Filename Prefix", "Detail"])

        for csv_path, i, end_row_num, name, positive_text, negative_text, notes, resolution, image_path in entries:
            label = f"[{csv_path.name} {i}/{end_row_num}]"

            if not positive_text and image_path and Path(image_path).is_file():
                try:
                    positive_text = describe_image(image_path)
                    print(f"{label} {name} has no prompt metadata - described the image via {VISION_MODEL} instead")
                except Exception as e:
                    print(f"{label} Skipping {name}: image description failed: {e}", file=sys.stderr)
                    log_writer.writerow([csv_path.name, name, "skipped", "", "", "", f"image description failed: {e}"])
                    continue

            if not positive_text:
                detail = notes or "no positive prompt text"
                print(f"{label} Skipping {name}: {detail}")
                log_writer.writerow([csv_path.name, name, "skipped", "", "", "", detail])
                continue

            lora_matches = select_loras(positive_text)
            lora_summary = ", ".join(f"{lora}@{strength}" for lora, strength in lora_matches) or "none"

            now = datetime.datetime.now()
            prefix = f"rerun/{now:%Y-%m-%d}/{now:%H%M%S_%f}_{Path(name).stem}"
            wf = build_workflow_for_row(
                template, positive_id, negative_id, save_ids,
                positive_text, negative_text, prefix, random_seed,
                lora_node_id, lora_matches, latent_node_id, resolution,
            )

            try:
                result = queue_prompt(server, wf, client_id)
            except urllib.error.URLError as e:
                print(f"{label} Failed to queue {name}: {e}", file=sys.stderr)
                log_writer.writerow([csv_path.name, name, "error", "", lora_summary, prefix, f"Failed to queue: {e}"])
                continue

            node_errors = result.get("node_errors")
            prompt_id = result.get("prompt_id")
            if node_errors:
                print(f"{label} {name}: node errors: {node_errors}")
                log_writer.writerow([csv_path.name, name, "error", prompt_id or "", lora_summary, prefix, json.dumps(node_errors)])
                continue

            print(f"{label} Queued {name} as prompt_id={prompt_id} (LoRAs: {lora_summary})")
            log_writer.writerow([csv_path.name, name, "queued", prompt_id, lora_summary, prefix, ""])

            log_file.flush()
            time.sleep(delay)

    print(f"\nAll prompts queued. Log written to: {log_path}")
    print("ComfyUI will process the queue in the background - check its window or output folder for results.")


def run(config_path):
    """JSON-config-driven entry point, same convention as
    run_test.run()/lora_test.run()/generate_prompt_variations.run() - a
    caller like the GUI can drive this with an explicit list of CSV paths
    (e.g. exactly the files a Variations-tab run just produced) without
    going through argparse or directory-discovery. Config file format:
        {
            "csv_paths": ["...", "..."],
            "workflow": "...",
            "server": "http://127.0.0.1:8000",   // optional
            "start_row": 3,                       // optional, single-path only
            "end_row": 7,                         // optional, single-path only
            "random_seed": false,                 // optional
            "delay": 0.2,                          // optional
            "log": "..."                           // optional
        }
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_batch(
        csv_paths=[Path(p) for p in config["csv_paths"]],
        workflow_path=Path(config["workflow"]).expanduser().resolve(),
        server=config.get("server", "http://127.0.0.1:8000"),
        start_row=config.get("start_row"),
        end_row=config.get("end_row"),
        random_seed=config.get("random_seed", False),
        delay=config.get("delay", 0.2),
        log_path=config.get("log"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csv_file",
        help="CSV file produced by extract_image_prompts.py, or a directory - in which case "
             "every *.csv file directly inside it is processed (rerun_log.csv excluded).",
    )
    parser.add_argument("start_row", type=int, nargs="?", default=None, help="First CSV row to rerun, 1-indexed inclusive (default: 1). Only valid when csv_file is a single file.")
    parser.add_argument("end_row", type=int, nargs="?", default=None, help="Last CSV row to rerun, 1-indexed inclusive (default: last row)")
    parser.add_argument(
        "--workflow",
        default=r"F:\Programs\ComfyFiles\user\default\workflows\krea2_basic_t2i.json",
        help="Workflow path used for every row. Resolved to an absolute path regardless of how it's given. "
             "Auto-converted via a headless browser if it's not already an API-format export - see "
             "'Getting the workflow template' above. (default: %(default)s)",
    )
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="ComfyUI server URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--random-seed", action="store_true", help="Randomize seed/noise_seed inputs for each row instead of reusing the template's seed")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between queuing each prompt (default: 0.2)")
    parser.add_argument("--log", default=None, help="Output log CSV path (default: <csv_file dir>/rerun_log.csv)")
    args = parser.parse_args()

    workflow_path = Path(args.workflow).expanduser().resolve()

    input_path = Path(args.csv_file)
    if input_path.is_dir():
        if args.start_row is not None or args.end_row is not None:
            print("Error: start_row/end_row require a single CSV file, not a directory (ambiguous across multiple CSVs)", file=sys.stderr)
            sys.exit(1)
        csv_paths = discover_csv_files(input_path)
        if not csv_paths:
            print(f"Error: no *.csv files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"{input_path} is a directory - found {len(csv_paths)} CSV file(s):")
        for p in csv_paths:
            print(f"  {p.name}")
    elif input_path.is_file():
        csv_paths = [input_path]
    else:
        print(f"Error: CSV file or directory not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run_batch(
        csv_paths=csv_paths,
        workflow_path=workflow_path,
        server=args.server,
        start_row=args.start_row,
        end_row=args.end_row,
        random_seed=args.random_seed,
        delay=args.delay,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
