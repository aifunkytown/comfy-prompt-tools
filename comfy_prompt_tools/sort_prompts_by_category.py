"""
Combines every *.csv file directly under a given directory (the current
directory by default) into category files, based on keyword categories and
subject count found in the "Positive Prompt" (and "Cleaned Prompt", if
present) column of each row.

The keyword categories themselves (which words trigger which category) live
in category_keywords.json next to this script, not in this file - see that
file for the current list. Categories are checked in the order they appear
there; a row lands in the first category whose "keywords" match (and whose
"exclude_keywords", if any, don't). Anything matching no category falls
through to the two built-in fallback categories:

    - everything else, single subject -> general_solo.csv
    - everything else, 2+ subjects    -> general_group.csv

An optional category_keywords.local.json next to the base file overrides it
the same way Claude Code's settings.local.json overrides settings.json: any
category there with the same "name" as a base category replaces it entirely,
and any new "name" is appended. It's gitignored, so it's the place for
personal/local-only keyword tweaks that shouldn't be committed.

Solo vs. group is decided by count_subjects(): first it looks for booru-style
counting tags in the Positive Prompt (solo, 1girl, 2girls, 1girl+1boy, duo,
trio, group, multiple, etc.) and sums them. If no such tags are found at all,
it falls back to scanning the Cleaned Prompt prose for group-indicating words
(two/three/several/pair/group/crowd, "another woman", "and her dog", etc.).
With no signal either way, it defaults to solo.

Source files are combined in filename order and left in place afterward (source
deletion is not automatic; remove the originals yourself once you've verified
the output files). rerun_log.csv and the output files themselves are always
excluded from the source scan.

Each category's existing <category>.csv, if any, is loaded first and its rows
carried forward - a re-run only ever *adds* newly-sorted rows into the
category files (subject to the same per-category dedupe as everything else),
it never drops what a previous run already sorted there.

Within each output category, rows are deduped on Positive Prompt text (same
convention as extract_image_prompts.py) - if two rows (whether from a source
file this run or already sitting in that category's existing output file)
land in the same category with identical Positive Prompt text, only the
first one seen is kept. Rows with no Positive Prompt text are never deduped
against each other. Duplicates are checked per-category, not globally, since
the same prompt legitimately appearing in two different categories (e.g. a
category from category_keywords.local.json vs. general) isn't a duplicate.

Usage:
    # Scans the current directory for source CSVs, writes category CSVs
    # into that same directory:
    python sort_prompts_by_category.py

    # Scans a specific directory instead:
    python sort_prompts_by_category.py F:\\Programs\\ComfyFiles\\output

    # Writes the category CSVs somewhere other than the source directory:
    python sort_prompts_by_category.py F:\\Programs\\ComfyFiles\\output --output-dir F:\\Programs\\ComfyFiles\\output\\sorted
"""

import argparse
import csv
import re
from pathlib import Path

try:
    from local_config import load_named_list, load_list  # run directly: python sort_prompts_by_category.py
except ImportError:
    from comfy_prompt_tools.local_config import load_named_list, load_list  # imported as a package

CONFIG_PATH = Path(__file__).resolve().parent / "category_keywords.json"
COUNT_TAG_NOUNS_PATH = Path(__file__).resolve().parent / "count_tag_nouns.json"

FALLBACK_CATEGORIES = ("general_solo", "general_group")

CATEGORIES = load_named_list(CONFIG_PATH, "categories", "name")
CATEGORY_NAMES = [cat["name"] for cat in CATEGORIES] + list(FALLBACK_CATEGORIES)

EXCLUDE = {"rerun_log.csv", "general.csv"} | {f"{name}.csv" for name in CATEGORY_NAMES}

# Booru-style counting-tag nouns (e.g. the "girls" in "2girls") - like
# CATEGORIES, extended by a gitignored count_tag_nouns.local.json next to
# count_tag_nouns.json for personal additions.
_COUNT_TAG_NOUNS = load_list(COUNT_TAG_NOUNS_PATH, "nouns")
COUNT_TAG_RE = re.compile(r"\b(\d+)\s*(" + "|".join(_COUNT_TAG_NOUNS) + r")\b")
SOLO_TAG_RE = re.compile(r"\bsolo\b")
DUO_TAG_RE = re.compile(r"\bduo\b")
TRIO_TAG_RE = re.compile(r"\btrio\b")
GROUP_TAG_RE = re.compile(
    r"\bgroup\b|\bmultiple\s+(girls?|boys?|people|women|men|others?|characters?|figures?)\b"
)

# Number/quantity words require a person-noun right after them, since bare
# words like "two" or "multiple" show up constantly in unrelated contexts in
# these prompts (e.g. "two braids hair style", "multiple orgasms").
PROSE_GROUP_RE = re.compile(
    r"\b(two|three|four|five|several)\s+(women|men|girls|boys|people|figures|friends|companions|characters|others)\b"
    r"|\bmultiple\s+(women|men|girls|boys|people|figures|friends|companions|characters|others)\b"
    r"|\bgroup of\b"
    r"|\bcrowd\b"
    r"|\btrio\b"
    r"|\bcouple\b"
    r"|\beach other\b"
    r"|\bboth (?:women|men|girls|boys|of them)\b"
    r"|\banother (?:woman|man|girl|boy|person|figure)\b"
    r"|\ba second (?:woman|man|girl|boy|person)\b"
    r"|\b(?:and|with) (?:her|his|their) (?:dog|cat|wolf|fox|pet|companion|friend|partner|boyfriend|girlfriend)\b"
)


def count_subjects_from_tags(text):
    """Return (total_count, found_any) from booru-style counting tags."""
    total = 0
    found_any = False
    for m in COUNT_TAG_RE.finditer(text):
        total += int(m.group(1))
        found_any = True
    if SOLO_TAG_RE.search(text):
        total = max(total, 1)
        found_any = True
    if DUO_TAG_RE.search(text):
        total = max(total, 2)
        found_any = True
    if TRIO_TAG_RE.search(text):
        total = max(total, 3)
        found_any = True
    if GROUP_TAG_RE.search(text):
        total = max(total, 2)
        found_any = True
    return total, found_any


def is_group(row):
    positive = (row.get("Positive Prompt") or "").lower()
    total, found_any = count_subjects_from_tags(positive)
    if found_any:
        return total >= 2

    # No booru-style tags at all; fall back to prose in Cleaned Prompt
    # (or Positive Prompt if there's no Cleaned Prompt column/value).
    prose = (row.get("Cleaned Prompt") or "").lower() or positive
    return bool(PROSE_GROUP_RE.search(prose))


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_bucket_from_existing_output(category, output_dir, buckets, seen_prompts, fieldnames):
    """Load a category's existing <category>.csv (if any) into buckets/
    seen_prompts before any new rows are added, so writing the file back out
    later merges with what a previous run already sorted there instead of
    replacing it."""
    existing_path = output_dir / f"{category}.csv"
    if not existing_path.is_file():
        return
    existing_fields, existing_rows = read_csv(existing_path)
    for field in existing_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    for row in existing_rows:
        positive = (row.get("Positive Prompt") or "").strip()
        label = row.get("File Name") or row.get("File Path") or "(unnamed row)"
        if positive:
            seen_prompts[category][positive] = label
        buckets[category].append(row)


def categorize(row):
    text = " ".join(
        (row.get(col) or "") for col in ("Positive Prompt", "Cleaned Prompt") if col in row
    ).lower()
    for cat in CATEGORIES:
        if any(k in text for k in cat["keywords"]) and not any(
            k in text for k in cat.get("exclude_keywords", ())
        ):
            return cat["name"]
    return "general_group" if is_group(row) else "general_solo"


def run(directory=".", output_dir=None):
    directory = Path(directory)
    output_dir = Path(output_dir) if output_dir else directory

    csv_files = sorted(
        p for p in directory.glob("*.csv") if p.name not in EXCLUDE
    )
    print(f"Found {len(csv_files)} source CSV file(s) in {directory}")

    if not csv_files:
        names = "/".join(f"{name}.csv" for name in CATEGORY_NAMES)
        print(f"No source CSV files found in {directory}; leaving existing {names} untouched.")
        return

    fieldnames = []
    buckets = {name: [] for name in CATEGORY_NAMES}
    seen_prompts = {category: {} for category in buckets}
    duplicates_skipped = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORY_NAMES:
        seed_bucket_from_existing_output(category, output_dir, buckets, seen_prompts, fieldnames)

    for path in csv_files:
        file_fields, rows = read_csv(path)
        for field in file_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            category = categorize(row)
            positive = (row.get("Positive Prompt") or "").strip()
            label = row.get("File Name") or row.get("File Path") or "(unnamed row)"

            if positive:
                if positive in seen_prompts[category]:
                    print(
                        f"Skipping duplicate prompt in {category}: {label} "
                        f"(same Positive Prompt as {seen_prompts[category][positive]})"
                    )
                    duplicates_skipped += 1
                    continue
                seen_prompts[category][positive] = label

            buckets[category].append(row)

    for category, rows in buckets.items():
        out_path = output_dir / f"{category}.csv"
        write_csv(out_path, fieldnames, rows)
        print(f"{category}.csv: {len(rows)} rows")

    print(f"Skipped {duplicates_skipped} duplicate row(s) across all categories.")
    print(f"Source file(s) left in place ({len(csv_files)}); delete them manually once verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "directory", nargs="?", default=".",
        help="Directory to scan for source *.csv files (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Directory to write the category CSV files into (default: same as the source directory)",
    )
    args = parser.parse_args()
    run(args.directory, args.output_dir)


if __name__ == "__main__":
    main()
