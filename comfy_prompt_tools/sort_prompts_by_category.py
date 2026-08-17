"""
Combines every *.csv file directly under F:\\Programs\\ComfyFiles\\output into 4
category files, based on keywords and subject count found in the "Positive
Prompt" (and "Cleaned Prompt", if present) column of each row:

    - contains "futa", "futanari", or "dickgirl" -> futa.csv
    - contains "furry" or "anthro"       -> furry.csv          (checked after futa)
    - everything else, single subject    -> general_solo.csv   (checked after futa/furry)
    - everything else, 2+ subjects       -> general_group.csv

Solo vs. group is decided by count_subjects(): first it looks for booru-style
counting tags in the Positive Prompt (solo, 1girl, 2girls, 1girl+1boy, duo,
trio, group, multiple, etc.) and sums them. If no such tags are found at all,
it falls back to scanning the Cleaned Prompt prose for group-indicating words
(two/three/several/pair/group/crowd, "another woman", "and her dog", etc.).
With no signal either way, it defaults to solo.

Source files are combined in filename order and left in place afterward (source
deletion is not automatic; remove the originals yourself once you've verified
the 4 output files). rerun_log.csv and the 4 output files themselves are always
excluded from the source scan.

Usage:
    python sort_prompts_by_category.py
"""

import csv
import re
from pathlib import Path

OUTPUT_DIR = Path(r"F:\Programs\ComfyFiles\output")
EXCLUDE = {
    "rerun_log.csv",
    "futa.csv",
    "furry.csv",
    "general.csv",
    "general_solo.csv",
    "general_group.csv",
}

FUTA_KEYWORDS = ("futa", "futanari", "dickgirl")
FURRY_KEYWORDS = ("furry", "anthro")

COUNT_TAG_RE = re.compile(
    r"\b(\d+)\s*(girls?|boys?|others?|futas?|anthros?|furr(?:y|ies)|females?|males?|people|persons?)\b"
)
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


def categorize(row):
    text = " ".join(
        (row.get(col) or "") for col in ("Positive Prompt", "Cleaned Prompt") if col in row
    ).lower()
    if any(k in text for k in FUTA_KEYWORDS):
        return "futa"
    if any(k in text for k in FURRY_KEYWORDS):
        return "furry"
    return "general_group" if is_group(row) else "general_solo"


def main():
    csv_files = sorted(
        p for p in OUTPUT_DIR.glob("*.csv") if p.name not in EXCLUDE
    )
    print(f"Found {len(csv_files)} source CSV file(s)")

    if not csv_files:
        print("No source CSV files found; leaving existing futa.csv/furry.csv/general_solo.csv/general_group.csv untouched.")
        return

    fieldnames = []
    buckets = {"futa": [], "furry": [], "general_solo": [], "general_group": []}

    for path in csv_files:
        file_fields, rows = read_csv(path)
        for field in file_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            buckets[categorize(row)].append(row)

    for category, rows in buckets.items():
        out_path = OUTPUT_DIR / f"{category}.csv"
        write_csv(out_path, fieldnames, rows)
        print(f"{category}.csv: {len(rows)} rows")

    print(f"Source file(s) left in place ({len(csv_files)}); delete them manually once verified.")


if __name__ == "__main__":
    main()
