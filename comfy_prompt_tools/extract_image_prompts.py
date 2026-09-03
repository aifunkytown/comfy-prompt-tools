"""
Scan a directory (and all sub-directories) for images and extract the
generation prompts embedded in their metadata into a CSV file.

Supports:
  - ComfyUI PNGs (embedded "prompt" API-graph JSON in the PNG text chunks)
  - Automatic1111 / A1111-style PNGs (embedded "parameters" text chunk)
  - InvokeAI PNGs (embedded "invokeai_metadata" JSON text chunk)
  - JPEG/WEBP images that carry the same info via EXIF UserComment

Each row also gets a SHA-256 hash of its extracted prompt text (positive +
negative), for quick manual duplicate spotting. Any image whose Positive
Prompt text exactly matches one already seen (within the same folder) is
treated as a duplicate and skipped - only the first image using that
prompt is written to the CSV; the Negative Prompt and other fields are
ignored when checking for duplicates. Images with no positive prompt text
extracted from metadata are still written, with an empty Positive Prompt -
this lets clean_prompts.py fall back to describing the image directly via
a vision-capable Ollama model (using this row's File Path) instead of
having nothing to work with at all. That empty-prompt case is never
treated as a duplicate of another empty-prompt row.

One CSV is written per folder that directly contains images, named after
that folder (e.g. images in a folder called "2026-05-19" produce
"2026-05-19-prompts.csv"), saved into that same folder (the directory being
scanned) by default - use -o/--output-dir to redirect every CSV to one
custom location instead.

If a folder's CSV already exists, new rows are appended to it rather than
overwriting it - existing rows (and any extra columns already in the file,
e.g. a "Cleaned Prompt" column added by clean_prompts.py) are left alone,
and the existing rows' prompts are included in the duplicate check so
re-running on the same folder won't re-add images already recorded there.

Usage:
    python extract_image_prompts.py [directory] [-o output_dir]

Run with no arguments to scan the current directory (and its
sub-directories).

Example:
    python extract_image_prompts.py "F:\\Programs\\ComfyFiles\\output\\Prompts"
    python extract_image_prompts.py "F:\\Programs\\ComfyFiles\\output\\Prompts" -o "F:\\Prompts CSVs"

Requires:
    pip install Pillow
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# Common custom-node field names that hold a literal prompt string, beyond the
# usual CLIPTextEncode "text" input - e.g. Impact Pack's wildcard processor uses
# "populated_text" (the fully resolved text, vs. "wildcard_text" which may still
# contain unresolved wildcard syntax), and WeiLinPromptUI uses "positive".
LITERAL_TEXT_KEYS = ("text", "positive", "populated_text", "string")

# Nodes whose job is joining two text fragments together - resolved by
# recursively resolving each side and joining with the (literal) delimiter,
# rather than just picking whichever side happens to be a literal.
JOIN_NODE_SPECS = {
    "JoinStrings": ("string1", "string2", "delimiter"),
    "StringConcatenate": ("string_a", "string_b", "delimiter"),
}


def _resolve_link_value(nodes, value, visited, depth):
    """Resolve an inputs[...] value that's either already a literal string or a
    [node_id, output_slot] link to another node."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 2:
        return _resolve_node_text(nodes, value[0], visited, depth + 1)
    return None


def _resolve_node_text(nodes, node_id, visited=None, depth=0, max_depth=15):
    """Best-effort resolution of the literal prompt text feeding a node, tracing
    back through intermediate conditioning-modifier / string-building nodes
    (e.g. ConditioningKrea2Rebalance, ConditioningConcat, JoinStrings) until a
    literal string is found. Not every custom node's text-producing field is
    understood - this covers the common patterns, not every possible node."""
    if visited is None:
        visited = set()
    key = str(node_id)
    if key in visited or depth > max_depth:
        return None
    visited = visited | {key}

    node = nodes.get(key)
    if not node:
        return None
    inputs = node.get("inputs", {})
    class_type = node.get("class_type", "")

    if class_type in JOIN_NODE_SPECS:
        key_a, key_b, key_delim = JOIN_NODE_SPECS[class_type]
        delim = inputs.get(key_delim)
        delim = delim if isinstance(delim, str) else ""
        part_a = _resolve_link_value(nodes, inputs.get(key_a), visited, depth)
        part_b = _resolve_link_value(nodes, inputs.get(key_b), visited, depth)
        parts = [p for p in (part_a, part_b) if p]
        return delim.join(parts) if parts else None

    for field in LITERAL_TEXT_KEYS:
        val = inputs.get(field)
        if isinstance(val, str) and val.strip():
            return val

    for val in inputs.values():
        if isinstance(val, list) and len(val) == 2:
            result = _resolve_node_text(nodes, val[0], visited, depth + 1, max_depth)
            if result:
                return result
    return None


def extract_comfyui_prompts(prompt_json_text):
    """Parse a ComfyUI 'prompt' API-graph JSON string into (positive, negative) text."""
    nodes = json.loads(prompt_json_text)

    for node in nodes.values():
        if "KSampler" in node.get("class_type", ""):
            inputs = node.get("inputs", {})
            pos_link = inputs.get("positive")
            neg_link = inputs.get("negative")
            positive = _resolve_node_text(nodes, pos_link[0]) if isinstance(pos_link, list) and pos_link else None
            negative = _resolve_node_text(nodes, neg_link[0]) if isinstance(neg_link, list) and neg_link else None
            if positive or negative:
                return positive, negative

    # Fallback: no KSampler found (or no linked text) - just grab CLIPTextEncode nodes in
    # order. Deliberately literal-only here (no deep resolution): without a KSampler to
    # explicitly designate which link is "positive", there's no reliable way to know
    # which of possibly several unrelated CLIPTextEncode nodes (e.g. in a multi-stage
    # detailer/upscale pipeline) is "the" prompt, so guessing via a multi-hop trace risks
    # confidently returning the wrong (or an unrelated fragment of) text.
    texts = [
        node.get("inputs", {}).get("text")
        for node in nodes.values()
        if node.get("class_type") == "CLIPTextEncode" and isinstance(node.get("inputs", {}).get("text"), str)
    ]
    texts = [t for t in texts if t]
    if texts:
        positive = texts[0]
        negative = texts[1] if len(texts) > 1 else None
        return positive, negative

    return None, None


def parse_a1111_parameters(params_text):
    """Parse an A1111-style 'parameters' text blob into (positive, negative, other)."""
    if "Negative prompt:" in params_text:
        positive, rest = params_text.split("Negative prompt:", 1)
    else:
        positive, rest = params_text, ""

    negative = rest
    other = ""
    for marker in ("\nSteps:", "Steps:"):
        if marker in rest:
            negative, other = rest.split(marker, 1)
            other = "Steps:" + other
            break

    return positive.strip(), negative.strip(), other.strip()


def parse_invokeai_metadata(metadata_json_text):
    """Parse InvokeAI's flat 'invokeai_metadata' JSON into (positive, negative, other)."""
    data = json.loads(metadata_json_text)

    positive = (data.get("positive_prompt") or "").strip()
    negative = (data.get("negative_prompt") or "").strip()

    other_parts = []
    if data.get("steps") is not None:
        other_parts.append(f"Steps: {data['steps']}")
    if data.get("cfg_scale") is not None:
        other_parts.append(f"CFG scale: {data['cfg_scale']}")
    if data.get("seed") is not None:
        other_parts.append(f"Seed: {data['seed']}")
    if data.get("width") and data.get("height"):
        other_parts.append(f"Size: {data['width']}x{data['height']}")
    model_name = (data.get("model") or {}).get("name")
    if model_name:
        other_parts.append(f"Model: {model_name}")
    other = ", ".join(other_parts)

    return positive, negative, other


def get_exif_user_comment(image):
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None

    # UserComment (0x9286) normally lives in the Exif sub-IFD (pointed to by
    # ExifOffset, 0x8769), not the top-level IFD0 dict - check both.
    raw = exif.get(0x9286)
    if raw is None:
        try:
            sub_ifd = exif.get_ifd(0x8769)
        except Exception:
            sub_ifd = {}
        raw = sub_ifd.get(0x9286)
    if not raw:
        return None

    if isinstance(raw, bytes):
        unicode_prefix = b"UNICODE\x00\x00"
        ascii_prefix = b"ASCII\x00\x00\x00"
        if raw.startswith(unicode_prefix):
            try:
                return raw[len(unicode_prefix):].decode("utf-16-le", errors="ignore").strip("\x00").strip()
            except Exception:
                return None
        if raw.startswith(ascii_prefix):
            raw = raw[len(ascii_prefix):]
        try:
            return raw.decode("utf-8", errors="ignore").strip("\x00").strip()
        except Exception:
            return None
    return str(raw)


def hash_prompt(positive, negative):
    """Return the SHA-256 hex digest of the combined positive+negative prompt text."""
    combined = (positive.strip() + "\x00" + negative.strip()).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def extract_row(image_path):
    """Return (positive, negative, other_params, source_format, notes) for one image file.

    Some files carry multiple metadata sources at once (e.g. Civitai-saved PNGs
    can have "prompt", "workflow", AND an A1111-style "parameters" block all
    together). "prompt" is tried first since it cleanly separates positive from
    negative, but if it parses without error and still yields no usable
    *positive* text (e.g. a graph using broadcast/set-node routing our
    resolver can't trace for the positive link, even if the negative link
    happens to be a plain literal), this falls through to the other sources
    instead of giving up - a positive prompt is required to use a result at
    all, since that's what everything downstream (dedup, CSV output) keys on."""
    try:
        with Image.open(image_path) as img:
            info = img.info
            fallback_note = None

            if "prompt" in info:
                try:
                    positive, negative = extract_comfyui_prompts(info["prompt"])
                    if positive:
                        return positive, negative or "", "", "ComfyUI", ""
                except Exception as e:
                    fallback_note = f"Failed to parse ComfyUI prompt JSON: {e}"

            if "parameters" in info:
                positive, negative, other = parse_a1111_parameters(info["parameters"])
                if positive.strip():
                    return positive, negative, other, "A1111", ""

            if "invokeai_metadata" in info:
                try:
                    positive, negative, other = parse_invokeai_metadata(info["invokeai_metadata"])
                    if positive.strip():
                        return positive, negative, other, "InvokeAI", ""
                except Exception as e:
                    fallback_note = f"Failed to parse invokeai_metadata JSON: {e}"

            comment = get_exif_user_comment(img)
            if comment:
                try:
                    positive, negative = extract_comfyui_prompts(comment)
                    if positive:
                        return positive, negative or "", "", "ComfyUI (EXIF)", ""
                except Exception:
                    pass
                if "Negative prompt:" in comment or "Steps:" in comment:
                    positive, negative, other = parse_a1111_parameters(comment)
                    if positive.strip():
                        return positive, negative, other, "A1111 (EXIF)", ""
                if comment.strip():
                    return comment, "", "", "EXIF UserComment", ""

            return "", "", "", "", fallback_note or "No prompt metadata found"

    except Exception as e:
        return "", "", "", "", f"Error reading image: {e}"


STANDARD_FIELDNAMES = [
    "File Name", "File Path", "Positive Prompt", "Negative Prompt",
    "Other Parameters", "Source Format", "Notes", "Prompt Hash (SHA-256)",
]


def write_folder_csv(folder, image_paths, output_dir=None):
    """Write (or append to) <folder name>-prompts.csv, deduped within that folder on
    Positive Prompt text - an image with no positive prompt is still written, with
    that column empty, rather than skipped (see the module docstring). Saved into
    output_dir if given, otherwise directly into folder itself (the directory being
    scanned). If the CSV already exists, its rows are kept, its own column layout is
    preserved (so extra columns like clean_prompts.py's "Cleaned Prompt" survive),
    and its existing prompts seed the duplicate check so re-running doesn't re-add
    images already recorded there."""
    output_path = (output_dir or folder) / f"{folder.name}-prompts.csv"
    file_exists = output_path.is_file()

    seen_positive_prompts = {}
    fieldnames = STANDARD_FIELDNAMES
    if file_exists:
        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or STANDARD_FIELDNAMES
            for row in reader:
                positive = (row.get("Positive Prompt") or "").strip()
                if positive:
                    seen_positive_prompts[positive] = f"an existing row in {output_path.name}"

    rows_written = 0
    duplicates_skipped = 0
    no_metadata_count = 0

    with open(output_path, "a" if file_exists else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        for image_path in image_paths:
            positive, negative, other, source_format, notes = extract_row(image_path)

            if positive.strip():
                if positive in seen_positive_prompts:
                    print(f"Skipping duplicate prompt: {image_path} (same Positive Prompt as {seen_positive_prompts[positive]})")
                    duplicates_skipped += 1
                    continue
                seen_positive_prompts[positive] = image_path
            else:
                print(f"No prompt metadata found, writing with an empty Positive Prompt: {image_path}")
                no_metadata_count += 1

            prompt_hash = hash_prompt(positive, negative)

            writer.writerow({
                "File Name": image_path.name,
                "File Path": str(image_path),
                "Positive Prompt": positive,
                "Negative Prompt": negative,
                "Other Parameters": other,
                "Source Format": source_format,
                "Notes": notes,
                "Prompt Hash (SHA-256)": prompt_hash,
            })
            rows_written += 1

    total_scanned = rows_written + duplicates_skipped
    action = f"Appended {rows_written} new row(s) to" if file_exists else "Wrote"
    print(
        f"{folder}: scanned {total_scanned} image(s), skipped {duplicates_skipped} duplicate(s), "
        f"{no_metadata_count} with no prompt metadata (written empty for image-based description). "
        f"{action} {output_path}"
    )


def extract_all(directory, output_dir=None):
    """Core logic behind main() - scan directory recursively for images,
    write/append one CSV per folder that directly contains any, and return
    the list of output CSV Paths touched (empty if no images were found).
    Callable directly by other scripts (e.g. extract_and_clean.py) without
    going through argparse."""
    root = Path(directory)
    if not root.is_dir():
        print(f"Error: directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    if output_dir:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found under {root}")
        return []

    by_folder = {}
    for image_path in image_paths:
        by_folder.setdefault(image_path.parent, []).append(image_path)

    seen_names = {}
    output_paths = []
    for folder in sorted(by_folder):
        if folder.name in seen_names:
            print(
                f"Warning: folder name '{folder.name}' seen at both {seen_names[folder.name]} and {folder} - "
                f"the second one will overwrite {folder.name}-prompts.csv",
                file=sys.stderr,
            )
        seen_names[folder.name] = folder
        write_folder_csv(folder, by_folder[folder], output_dir)
        output_paths.append((output_dir or folder) / f"{folder.name}-prompts.csv")

    return output_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan recursively (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to write every CSV file into (default: save each CSV into the folder it was scanned from). Created if it doesn't exist.",
    )
    args = parser.parse_args()

    extract_all(args.directory, args.output_dir)


if __name__ == "__main__":
    main()
