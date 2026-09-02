"""Shared helper for the base-config + gitignored .local override pattern
used across this project's scripts (mirrors Claude Code's settings.json /
settings.local.json): a `<name>.local.json` file next to `<name>.json` can
replace individual named entries and/or add new ones, without needing to
redefine the whole file. It's never committed - see .gitignore's *.local.json
rule - so it's the place for personal config (real LoRA filenames, NSFW
keyword lists, etc.) that shouldn't be in the public repo.
"""

import json
from pathlib import Path


def local_path_for(base_path: Path) -> Path:
    return base_path.with_name(base_path.stem + ".local" + base_path.suffix)


def merge_named_entries(base_entries, local_entries, key):
    """Local entries override the base entry with the same `key` value in
    place, preserving its position. Entries with a new `key` value are
    inserted before the base list, in local-file order - for callers where
    list order is a priority order (e.g. first-match-wins categories), this
    means local additions are checked before base ones, same as how a more
    specific override normally wins."""
    merged = list(base_entries)
    index_by_key = {entry[key]: i for i, entry in enumerate(merged)}
    new_entries = []
    for local_entry in local_entries:
        k = local_entry[key]
        if k in index_by_key:
            merged[index_by_key[k]] = local_entry
        else:
            new_entries.append(local_entry)
    return new_entries + merged


def load_named_list(base_path: Path, list_key: str, key: str):
    """Load base_path's JSON (a dict with a list under list_key), merging in
    local_path_for(base_path) the same way if it exists."""
    base = json.loads(base_path.read_text(encoding="utf-8"))[list_key]
    local_path = local_path_for(base_path)
    if local_path.is_file():
        local = json.loads(local_path.read_text(encoding="utf-8"))[list_key]
        base = merge_named_entries(base, local, key)
    return base


def load_local_text(base_path: Path, text_key: str) -> str:
    """Read local_path_for(base_path)'s JSON and return text_key, or "" if
    the local file doesn't exist."""
    local_path = local_path_for(base_path)
    if not local_path.is_file():
        return ""
    return json.loads(local_path.read_text(encoding="utf-8")).get(text_key, "")


def load_text(base_path: Path, text_key: str) -> str:
    """Read base_path's own (checked-in) JSON and return text_key - the
    base-file counterpart to load_local_text, for a single text value
    (e.g. a system prompt) instead of a named list."""
    return json.loads(base_path.read_text(encoding="utf-8"))[text_key]


def load_list(base_path: Path, list_key: str):
    """Load base_path's JSON list under list_key, extended (not
    override-by-name - just concatenated) by local_path_for(base_path)'s
    list under the same key if it exists."""
    items = list(json.loads(base_path.read_text(encoding="utf-8")).get(list_key, []))
    local_path = local_path_for(base_path)
    if local_path.is_file():
        items += json.loads(local_path.read_text(encoding="utf-8")).get(list_key, [])
    return items
