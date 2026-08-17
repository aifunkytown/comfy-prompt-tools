"""
Deduplicates rows in every *.csv file directly under F:\\Programs\\ComfyFiles\\output,
based solely on the "Positive Prompt" column. Within each file independently, the
first row for a given Positive Prompt is kept and any later row with the same
Positive Prompt is dropped. Other columns are ignored when comparing rows.

Usage:
    python dedupe_prompts_csv.py
"""

import csv
import os
from pathlib import Path

OUTPUT_DIR = Path(r"F:\Programs\ComfyFiles\output")
PROMPT_COLUMN = "Positive Prompt"


def dedupe_file(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or PROMPT_COLUMN not in fieldnames:
        print(f"skip (no '{PROMPT_COLUMN}' column): {csv_path.name}")
        return

    seen = set()
    kept_rows = []
    for row in rows:
        prompt = row.get(PROMPT_COLUMN, "")
        if prompt in seen:
            continue
        seen.add(prompt)
        kept_rows.append(row)

    removed = len(rows) - len(kept_rows)
    if removed == 0:
        print(f"no duplicates: {csv_path.name} ({len(rows)} rows)")
        return

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    os.replace(tmp_path, csv_path)

    print(f"deduped: {csv_path.name} ({len(rows)} -> {len(kept_rows)} rows, removed {removed})")


def main():
    csv_files = sorted(OUTPUT_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV file(s) in {OUTPUT_DIR}")
    for csv_path in csv_files:
        dedupe_file(csv_path)


if __name__ == "__main__":
    main()
