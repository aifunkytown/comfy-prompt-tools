"""
Combines every *.csv file directly under a given directory (the current
directory by default) into category files, based on keyword categories and
subject count found in the "Cleaned Prompt" column of each row - then
further splits each category by Content Rating (see rate_prompts.py) into
one file per rating tier: <category>_g.csv, <category>_pg13.csv (PG and
PG-13 share a tier), <category>_r.csv, and <category>_x.csv (X and XXX
share a tier). Sorting always happens category first, then rating.

futa is the one exception - it's always explicit content, so a rating
split wouldn't distinguish anything useful there, and it's kept as a
single flat futa.csv instead of being tier-split like every other
category.

A row whose Content Rating isn't real ratable content - never rated at all
(blank), the model's response didn't parse (UNPARSED), a genuine content
policy flag (REVIEW), or the permanent refusal marker from clean_prompts.py/
rate_prompts.py (ERROR) - is never sorted into a category/tier file
regardless of category (futa included). All of these instead land in one
shared needs_review.csv, tagged with a "Sort Category" column recording
which category they came from - both so a human reviewing that file knows
where a fixed-up row belongs, and so a later run can read it back and
re-route it once its rating has actually been fixed, instead of it being
stuck there forever.

Only ever sorts off "Cleaned Prompt", never the raw "Positive Prompt" - a
row with no Cleaned Prompt yet (not run through clean_prompts.py) is left
uncategorized rather than guessed at from its unreliable raw tag-soup text:
it's logged and skipped (not written to any category file), staying in its
source CSV - which, like every source row, is left in place regardless (see
below) - so a later run picks it up once it actually has a Cleaned Prompt.

The keyword categories themselves (which words trigger which category) live
in category_keywords.json next to this script, not in this file - see that
file for the current list. Categories are checked in the order they appear
there; a row lands in the first category whose "keywords" match (and whose
"exclude_keywords", if any, don't). Anything matching no category falls
through to the two built-in fallback categories:

    - everything else, single subject -> general_solo (then rating-split)
    - everything else, 2+ subjects    -> general_group (then rating-split)

An optional category_keywords.local.json next to the base file overrides it
the same way Claude Code's settings.local.json overrides settings.json: any
category there with the same "name" as a base category replaces it entirely,
and any new "name" is appended. It's gitignored, so it's the place for
personal/local-only keyword tweaks that shouldn't be committed.

Solo vs. group is decided by count_subjects(): first it looks for booru-style
counting tags in the Cleaned Prompt (solo, 1girl, 2girls, 1girl+1boy, duo,
trio, group, multiple, etc. - rare in a natural-language rewrite, but checked
for robustness) and sums them. If no such tags are found at all, it falls
back to scanning the same Cleaned Prompt's prose for group-indicating words
(two/three/several/pair/group/crowd, "another woman", "and her dog", etc.).
With no signal either way, it defaults to solo.

Source files are combined in filename order and left in place afterward (source
deletion is not automatic; remove the originals yourself once you've verified
the output files). rerun_log.csv, needs_review.csv, and every category's
output files (its tier files, or flat file for futa) are always excluded
from the source scan - but see below for running this against an
already-sorted directory.

Each category's existing output (its tier files, or flat file for futa),
plus its rows currently sitting in needs_review.csv, are loaded first and
carried forward - a re-run only ever *adds* newly-sorted rows on top
(subject to the same per-category dedupe as everything else), it never
drops what a previous run already sorted there. This also means it's safe
to point this script directly at an already-sorted directory (e.g. to
apply the rating split to a directory that predates it): a legacy flat
<category>.csv left over from before rating-splitting existed is read the
same way and then removed once its content has been folded into the new
tier files, rather than sitting there excluded and effectively invisible.

Within each output category, rows are deduped on Positive Prompt text (same
convention as extract_image_prompts.py) - if two rows (whether from a source
file this run or already sitting in that category's existing output file)
land in the same category with identical Positive Prompt text, only the
first one seen is kept. Rows with no Positive Prompt text are never deduped
against each other. Duplicates are checked per-category, not globally, since
the same prompt legitimately appearing in two different categories (e.g. a
category from category_keywords.local.json vs. general) isn't a duplicate.

Usage:
    # Scans the current directory for source CSVs, writes category/tier CSVs
    # into that same directory:
    python sort_prompts_by_category.py

    # Scans a specific directory instead:
    python sort_prompts_by_category.py F:\\Programs\\ComfyFiles\\output

    # Writes the category/tier CSVs somewhere other than the source directory:
    python sort_prompts_by_category.py F:\\Programs\\ComfyFiles\\output --output-dir F:\\Programs\\ComfyFiles\\output\\sorted

    # Re-derive the rating split for a directory that's already sorted
    # (e.g. one last sorted before rating-splitting existed) - no new
    # source CSVs needed, this reads back what's already there:
    python sort_prompts_by_category.py F:\\Programs\\ComfyFiles\\output\\sorted
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

# futa is always explicit content - a rating split wouldn't distinguish
# anything useful there, so it's the one category kept as a single flat
# file instead of being split into per-tier files like every other
# category (see split_by_rating()/run()). Rows needing attention (see
# rating_tier()) are still pulled out of it into NEEDS_REVIEW_FILE, same as
# any other category - "always explicit" is a statement about content
# severity, not about whether a row was successfully processed.
FUTA_CATEGORY = "futa"

# A category's rows are split by Content Rating into one file per tier
# (e.g. furry_g.csv, furry_pg13.csv, furry_r.csv, furry_x.csv) instead of
# one flat <category>.csv - PG and PG-13 share a tier, as do X and XXX,
# since the four-way G/PG-13/R/X split is what's actually useful for
# picking a folder to browse, not the full six-point rating scale.
RATING_TIER_MAP = {
    "G": "g",
    "PG": "pg13", "PG-13": "pg13",
    "R": "r",
    "X": "x", "XXX": "x",
}
RATING_TIERS = ("g", "pg13", "r", "x")

# Rows whose Content Rating isn't real ratable content - never rated yet
# (blank), the model's response didn't parse (UNPARSED), a genuine content
# policy flag (REVIEW), or the permanent refusal marker from clean_prompts.
# py/rate_prompts.py (ERROR) - all land here instead of any category/tier
# file, regardless of category (futa included). A "Sort Category" column
# records which category each row came from, both so a human reviewing
# this file knows where a fixed-up row belongs, and so a later run can
# read it back and re-route the row once its rating has actually been
# fixed (see load_needs_review_rows()) instead of it being stuck here
# forever.
NEEDS_REVIEW_FILE = "needs_review.csv"
SORT_CATEGORY_COLUMN = "Sort Category"


def rating_tier(rating):
    """The RATING_TIERS suffix for a Content Rating value, or None if the
    row belongs in NEEDS_REVIEW_FILE instead (see NEEDS_REVIEW_FILE's
    comment above) - covers a blank/missing rating the same way, since
    "never rated" isn't ratable content either."""
    return RATING_TIER_MAP.get((rating or "").strip().upper())


_TIER_OUTPUT_NAMES = {
    f"{name}_{tier}.csv"
    for name in CATEGORY_NAMES if name != FUTA_CATEGORY
    for tier in RATING_TIERS
}
EXCLUDE = (
    {"rerun_log.csv", "general.csv", NEEDS_REVIEW_FILE}
    | {f"{name}.csv" for name in CATEGORY_NAMES}
    | _TIER_OUTPUT_NAMES
)

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
    # Cleaned Prompt only - see categorize()'s docstring on why this script
    # never sorts off the raw, uncleaned Positive Prompt.
    prose = (row.get("Cleaned Prompt") or "").lower()
    total, found_any = count_subjects_from_tags(prose)
    if found_any:
        return total >= 2
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


def load_existing_category_rows(category, output_dir):
    """All rows already sorted into `category`'s existing output file(s):
    its per-rating-tier files (or, for futa, its one flat file), plus a
    legacy flat <category>.csv if one is still sitting there from before
    this script split output by rating. Reading (and run() then removing)
    that legacy file is what lets pointing this script at an
    already-sorted directory migrate it into the new tier-file layout
    instead of the old flat files just becoming invisible, excluded source
    material - see the module docstring."""
    if category == FUTA_CATEGORY:
        paths = [output_dir / "futa.csv"]
    else:
        paths = [output_dir / f"{category}_{tier}.csv" for tier in RATING_TIERS]
        legacy_flat = output_dir / f"{category}.csv"
        if legacy_flat.is_file():
            paths.append(legacy_flat)

    rows = []
    for path in paths:
        if path.is_file():
            _, file_rows = read_csv(path)
            rows.extend(file_rows)
    return rows


def load_needs_review_rows(output_dir):
    """{category: [rows]} for every row currently sitting in
    NEEDS_REVIEW_FILE, keyed by its stored Sort Category column (stripped
    from each row dict here, so it doesn't leak into a bucket and then a
    normal category/tier file's fieldnames) - read back in like
    load_existing_category_rows() so a re-run re-evaluates a since-fixed
    rating (e.g. an UNPARSED row that got successfully re-rated) and routes
    it back into its proper category/tier, instead of it being stuck in
    NEEDS_REVIEW_FILE forever once it lands there."""
    path = output_dir / NEEDS_REVIEW_FILE
    by_category = {}
    if not path.is_file():
        return by_category
    _, rows = read_csv(path)
    for row in rows:
        category = row.pop(SORT_CATEGORY_COLUMN, None) or "general_solo"
        by_category.setdefault(category, []).append(row)
    return by_category


def split_by_rating(rows):
    """rows split by rating_tier() into (tier_groups, normal_rows,
    flagged_rows): tier_groups is {tier: [rows]} for RATING_TIERS: a normal
    category writes one file per tier from this. normal_rows is every row
    that got a real tier (the same rows as tier_groups, flattened) - futa
    writes this to its one flat file instead, since it skips the tier
    split (see FUTA_CATEGORY). flagged_rows is everything else (see
    rating_tier()) - every category, futa included, routes these into
    NEEDS_REVIEW_FILE instead of any category/tier file."""
    tier_groups = {tier: [] for tier in RATING_TIERS}
    normal_rows = []
    flagged_rows = []
    for row in rows:
        tier = rating_tier(row.get("Content Rating"))
        if tier is None:
            flagged_rows.append(row)
        else:
            tier_groups[tier].append(row)
            normal_rows.append(row)
    return tier_groups, normal_rows, flagged_rows


def categorize(row):
    """Cleaned Prompt only - never the raw Positive Prompt. A row with no
    Cleaned Prompt yet hasn't been through clean_prompts.py, and its raw
    tag-soup text is unreliable for both keyword matching (booru tags don't
    read the same as the keyword lists were written against) and the
    is_group() subject count - such a row is skipped by run() before this
    is ever called (see there), rather than guessed at from the base
    prompt."""
    text = (row.get("Cleaned Prompt") or "").lower()
    for cat in CATEGORIES:
        if any(k in text for k in cat["keywords"]) and not any(
            k in text for k in cat.get("exclude_keywords", ())
        ):
            return cat["name"]
    return "general_group" if is_group(row) else "general_solo"


def run(directory=".", output_dir=None):
    directory = Path(directory)
    output_dir = Path(output_dir) if output_dir else directory
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(
        p for p in directory.glob("*.csv") if p.name not in EXCLUDE
    )
    print(f"Found {len(csv_files)} source CSV file(s) in {directory}")

    fieldnames = []
    buckets = {name: [] for name in CATEGORY_NAMES}
    seen_prompts = {category: {} for category in buckets}
    duplicates_skipped = 0
    uncleaned_skipped = 0

    # Phase 1 (category): seed every category's bucket from whatever it's
    # already sorted into on disk - its tier files (or flat file for futa),
    # a legacy pre-tier-split flat file if one is still there, and any of
    # its rows currently parked in NEEDS_REVIEW_FILE (re-evaluated below in
    # case a since-fixed rating now gives them a real tier) - then add
    # newly-categorized rows from csv_files on top. This always runs, even
    # with zero new source files, so pointing this script at an
    # already-sorted directory (see the module docstring) still re-derives
    # the tier/needs-review split from what's already there.
    needs_review_by_category = load_needs_review_rows(output_dir)
    for category in CATEGORY_NAMES:
        existing_rows = load_existing_category_rows(category, output_dir)
        existing_rows += needs_review_by_category.pop(category, [])
        for row in existing_rows:
            for field in row.keys():
                if field not in fieldnames:
                    fieldnames.append(field)
            positive = (row.get("Positive Prompt") or "").strip()
            label = row.get("File Name") or row.get("File Path") or "(unnamed row)"
            if positive:
                seen_prompts[category][positive] = label
            buckets[category].append(row)

    # Any needs-review row whose stored Sort Category no longer matches a
    # current category (e.g. category_keywords.json dropped or renamed it)
    # falls back to general_solo rather than being silently lost.
    for stray_rows in needs_review_by_category.values():
        for row in stray_rows:
            positive = (row.get("Positive Prompt") or "").strip()
            label = row.get("File Name") or row.get("File Path") or "(unnamed row)"
            if positive:
                seen_prompts["general_solo"][positive] = label
            buckets["general_solo"].append(row)

    for path in csv_files:
        file_fields, rows = read_csv(path)
        for field in file_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            label = row.get("File Name") or row.get("File Path") or "(unnamed row)"

            if not (row.get("Cleaned Prompt") or "").strip():
                # Not run through clean_prompts.py yet - never sorted from
                # the raw Positive Prompt (see categorize()'s docstring).
                # Left uncategorized (not written to any bucket/category
                # file) so it's picked up once it actually has a Cleaned
                # Prompt - the row stays in this source file regardless,
                # same as every other row (source files are always left in
                # place; see the module docstring).
                print(f"Skipping row with no Cleaned Prompt yet: {label}")
                uncleaned_skipped += 1
                continue

            category = categorize(row)
            positive = (row.get("Positive Prompt") or "").strip()

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

    # Phase 2 (rating): split each category's now-complete bucket by
    # Content Rating - one file per tier for a normal category, one flat
    # file for futa (see FUTA_CATEGORY) - and collect every row that isn't
    # real ratable content (rating_tier() is None: blank, UNPARSED, REVIEW,
    # or ERROR) into NEEDS_REVIEW_FILE instead, tagged with which category
    # it came from.
    all_flagged_rows = []
    for category in CATEGORY_NAMES:
        tier_groups, normal_rows, flagged_rows = split_by_rating(buckets[category])

        legacy_flat = output_dir / f"{category}.csv"
        if category == FUTA_CATEGORY:
            write_csv(output_dir / "futa.csv", fieldnames, normal_rows)
            print(f"futa.csv: {len(normal_rows)} rows")
        else:
            if legacy_flat.is_file():
                legacy_flat.unlink()
                print(f"Removed legacy {category}.csv (content now split by rating into {category}_<tier>.csv files)")
            for tier in RATING_TIERS:
                out_path = output_dir / f"{category}_{tier}.csv"
                write_csv(out_path, fieldnames, tier_groups[tier])
                print(f"{category}_{tier}.csv: {len(tier_groups[tier])} rows")

        for row in flagged_rows:
            row_copy = dict(row)
            row_copy[SORT_CATEGORY_COLUMN] = category
            all_flagged_rows.append(row_copy)

    needs_review_fieldnames = fieldnames + [SORT_CATEGORY_COLUMN]
    write_csv(output_dir / NEEDS_REVIEW_FILE, needs_review_fieldnames, all_flagged_rows)
    print(f"{NEEDS_REVIEW_FILE}: {len(all_flagged_rows)} rows")

    print(f"Skipped {duplicates_skipped} duplicate row(s) across all categories.")
    print(f"Skipped {uncleaned_skipped} row(s) with no Cleaned Prompt yet (run clean_prompts.py first, then re-run this).")
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
