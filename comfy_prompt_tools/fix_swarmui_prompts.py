"""
Fixes rows in already-extracted prompt CSVs whose "Positive Prompt" is a raw,
unparsed SwarmUI metadata JSON blob (a "sui_image_params"/"sui_extra_data"/
"sui_models" object) instead of the actual prompt text - the same bug
extract_image_prompts.py had before it learned to recognize SwarmUI's format
(see that script's parse_swarmui_metadata). Existing CSVs extracted before
that fix are stuck with the raw JSON dump baked into "Positive Prompt" and
need this one-time repair; new extractions no longer need it.

For each row whose Positive Prompt parses as that JSON shape, this replaces
Positive Prompt / Negative Prompt / Other Parameters / Source Format with the
correctly extracted values, and recomputes the Prompt Hash. If the row already
had a Cleaned Prompt, it's cleared out too - it was generated from the broken
raw-JSON input, so it's stale and should be regenerated (clean_prompts.py will
pick the row back up automatically next run, no --overwrite needed, since an
empty Cleaned Prompt is exactly what it treats as "not yet done"). Rows whose
Positive Prompt isn't that JSON shape are left completely untouched.

By default, fixed rows are corrected in place and stay in the same CSV. Pass
--move-to <path> to instead remove fixed rows from their source CSV entirely
and collect them into that one destination CSV (created if missing, appended
to - merging in any new columns - if it already exists, so multiple source
CSVs in one run all land in the same file). Each moved row gets a "Source CSV"
column recording which file it came from. This is meant for isolating rows
that need to be re-cleaned into their own file, separate from source CSVs
that shouldn't otherwise be disturbed.

Usage:
    python fix_swarmui_prompts.py <path-to-csv>
    python fix_swarmui_prompts.py <path-to-directory>   # every *.csv directly inside it
    python fix_swarmui_prompts.py <path-to-directory> --move-to needs_recleaning.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    from extract_image_prompts import parse_swarmui_metadata, hash_prompt  # run directly
except ImportError:
    from comfy_prompt_tools.extract_image_prompts import parse_swarmui_metadata, hash_prompt  # imported as a package

MOVE_SOURCE_COLUMN = "Source CSV"


def fix_rows(rows, fieldnames):
    """Fix every row in-place whose Positive Prompt is a raw SwarmUI JSON blob.
    Returns (fixed_count, cleaned_cleared_count)."""
    fixed = 0
    cleaned_cleared = 0
    for row in rows:
        swarmui = parse_swarmui_metadata(row.get("Positive Prompt") or "")
        if not swarmui:
            continue
        positive, negative, other = swarmui
        if not positive.strip():
            continue

        row["Positive Prompt"] = positive
        row["Negative Prompt"] = negative
        if "Other Parameters" in fieldnames:
            row["Other Parameters"] = other
        if "Source Format" in fieldnames:
            row["Source Format"] = "SwarmUI"
        if "Prompt Hash (SHA-256)" in fieldnames:
            row["Prompt Hash (SHA-256)"] = hash_prompt(positive, negative)
        fixed += 1

        if (row.get("Cleaned Prompt") or "").strip():
            row["Cleaned Prompt"] = ""
            cleaned_cleared += 1

    return fixed, cleaned_cleared


def write_csv(path: Path, fieldnames, rows):
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def append_to_move_file(move_path: Path, new_rows, new_fieldnames):
    """Merge new_rows into move_path, unioning in any fieldnames it doesn't
    already have (preserving existing column order, new ones appended at the
    end) so multiple differently-shaped source CSVs can all land in one file."""
    existing_fieldnames = []
    existing_rows = []
    if move_path.is_file():
        with open(move_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    combined_fieldnames = list(existing_fieldnames)
    for field in new_fieldnames:
        if field not in combined_fieldnames:
            combined_fieldnames.append(field)

    write_csv(move_path, combined_fieldnames, existing_rows + new_rows)


def fix_csv(csv_path: Path, move_path: Path = None):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "Positive Prompt" not in fieldnames:
        print(f"Skipping {csv_path}: no 'Positive Prompt' column found")
        return

    if not move_path:
        fixed, cleaned_cleared = fix_rows(rows, fieldnames)
        if fixed == 0:
            print(f"{csv_path.name}: no raw SwarmUI JSON rows found, left untouched")
            return
        write_csv(csv_path, fieldnames, rows)
        print(f"{csv_path.name}: fixed {fixed} row(s) in place, cleared {cleaned_cleared} stale Cleaned Prompt(s)")
        return

    # --move-to mode: fix a scratch copy of the rows, then only the ones that
    # actually changed get pulled out of the source and appended to move_path.
    remaining_rows = [dict(row) for row in rows]
    fixed, cleaned_cleared = fix_rows(remaining_rows, fieldnames)
    if fixed == 0:
        print(f"{csv_path.name}: no raw SwarmUI JSON rows found, left untouched")
        return

    moved_rows = []
    kept_rows = []
    for original, updated in zip(rows, remaining_rows):
        if updated["Positive Prompt"] != original.get("Positive Prompt"):
            updated[MOVE_SOURCE_COLUMN] = str(csv_path)
            moved_rows.append(updated)
        else:
            kept_rows.append(original)

    write_csv(csv_path, fieldnames, kept_rows)
    append_to_move_file(move_path, moved_rows, fieldnames + [MOVE_SOURCE_COLUMN])
    print(
        f"{csv_path.name}: moved {fixed} fixed row(s) (with {cleaned_cleared} stale Cleaned Prompt(s) "
        f"cleared) out to {move_path.name}, {len(kept_rows)} row(s) left behind"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="CSV file to fix, or a directory - in which case every *.csv file directly inside it is processed")
    parser.add_argument(
        "--move-to", default=None,
        help="Instead of fixing rows in place, remove them from their source CSV and collect them into this one destination CSV",
    )
    args = parser.parse_args()

    target = Path(args.path)
    if target.is_dir():
        csv_paths = sorted(p for p in target.glob("*.csv") if p.is_file())
        if not csv_paths:
            print(f"Error: no *.csv files found in {target}", file=sys.stderr)
            sys.exit(1)
    elif target.is_file():
        csv_paths = [target]
    else:
        print(f"Error: path not found: {target}", file=sys.stderr)
        sys.exit(1)

    move_path = Path(args.move_to).expanduser().resolve() if args.move_to else None

    for csv_path in csv_paths:
        if move_path and csv_path.resolve() == move_path:
            continue  # never process the destination file as if it were a source
        fix_csv(csv_path, move_path)


if __name__ == "__main__":
    main()
