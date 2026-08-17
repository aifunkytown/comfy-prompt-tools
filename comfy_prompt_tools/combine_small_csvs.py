"""
Combines small CSV files (fewer than 30 data rows) in F:\\Programs\\ComfyFiles\\output.

Files with 30+ rows are left untouched. All other (small) files are pooled together
regardless of adjacency, processed in filename (date) order, and grouped by
accumulating rows until the running total reaches 30 - at which point that group is
written out as a single combined file named after the newest (latest-date) file in
the group. Any trailing small files that never reach 30 are combined into a final
group of their own. The original files that get merged in (all but the one whose
name is reused) are deleted. rerun_log.csv is always excluded.

Usage:
    python combine_small_csvs.py
"""

import csv
import os
from pathlib import Path

OUTPUT_DIR = Path(r"F:\Programs\ComfyFiles\output")
THRESHOLD = 30
EXCLUDE = {"rerun_log.csv"}


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames, rows):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def flush_group(group):
    """group: list of (path, fieldnames, rows), in date order. Writes combined file
    under the newest path's name and deletes the other files in the group."""
    if not group:
        return
    if len(group) == 1:
        print(f"  no merge needed, only one small file in group: {group[0][0].name}")
        return

    newest_path = group[-1][0]
    fieldnames = []
    for _, file_fields, _ in group:
        for field in file_fields:
            if field not in fieldnames:
                fieldnames.append(field)

    combined_rows = []
    for path, _, rows in group:
        combined_rows.extend(rows)

    write_csv(newest_path, fieldnames, combined_rows)

    names = [p.name for p, _, _ in group]
    print(f"  combined {names} -> {newest_path.name} ({len(combined_rows)} rows)")

    for path, _, _ in group[:-1]:
        path.unlink()


def main():
    csv_files = sorted(
        p for p in OUTPUT_DIR.glob("*.csv") if p.name not in EXCLUDE
    )

    small_files = []
    for path in csv_files:
        fieldnames, rows = read_csv(path)
        if len(rows) >= THRESHOLD:
            print(f"skip (>= {THRESHOLD} rows): {path.name} ({len(rows)} rows)")
        else:
            small_files.append((path, fieldnames, rows))

    group = []
    group_total = 0
    for path, fieldnames, rows in small_files:
        group.append((path, fieldnames, rows))
        group_total += len(rows)
        if group_total >= THRESHOLD:
            flush_group(group)
            group = []
            group_total = 0

    flush_group(group)


if __name__ == "__main__":
    main()
