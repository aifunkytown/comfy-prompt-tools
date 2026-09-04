"""Shared helper for the base-config + gitignored .local override pattern
used across this project's scripts (mirrors Claude Code's settings.json /
settings.local.json): a `<name>.local.json` file next to `<name>.json` can
replace individual named entries and/or add new ones, without needing to
redefine the whole file. It's never committed - see .gitignore's *.local.json
rule - so it's the place for personal config (real LoRA filenames, NSFW
keyword lists, etc.) that shouldn't be in the public repo.
"""

import json
import re
import unicodedata
from pathlib import Path

# Curly/smart-typography characters a model sometimes emits despite being
# told not to - mapped to their plain-ASCII equivalents so a bad response
# never makes it into a CSV even if the system prompt is ignored. Also
# needed on Windows, independent of any CSV concern: a console using a
# legacy codepage (cp1252) raises UnicodeEncodeError on an unmapped
# character the moment a caller print()s it, which is worse than a lossy
# ASCII conversion - so every model response gets sanitized before either
# being written out or printed.
_ASCII_REPLACEMENTS = {
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-",                   # en dash
    "—": " - ",                 # em dash
    "…": "...",                 # ellipsis
    " ": " ",                   # non-breaking space
}


def sanitize_ascii(text: str) -> str:
    """Force text down to plain ASCII: map common smart-typography characters
    to their ASCII equivalents, decompose accented letters to their base form
    (e.g. "e" -> "e"), and drop anything left that still isn't ASCII (emoji,
    CJK, etc). Applied as a backstop after every model response, regardless
    of what the system prompt asked for."""
    if not text:
        return text
    for bad, good in _ASCII_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    return text.strip()


# Caps a single Ollama response - shared by clean_prompts.py and
# rate_prompts.py so neither script can be stalled by a degenerate
# repetition loop (a known local-model failure mode, e.g. endlessly
# repeating an escalating number). Without a cap, one bad row keeps
# generating for a very long time, which can push every row queued behind
# it past its own request timeout too - one bad row otherwise stalls the
# whole batch.
MAX_RESPONSE_TOKENS = 500

# How long a single Ollama request is allowed to run before urlopen gives
# up. Sized for MAX_RESPONSE_TOKENS's default of 500 - a caller that raises
# MAX_RESPONSE_TOKENS (e.g. for a reasoning model whose <think> block eats
# into the same budget) needs to raise this too, or a response that is
# genuinely still generating - not stuck - gets killed and logged as a
# timeout error before it ever finishes.
REQUEST_TIMEOUT_SECONDS = 180


def strip_thinking(raw_response: str) -> str:
    """Reasoning models (e.g. huihui_ai/qwen3-abliterated) generate inside
    a <think>...</think> block before their actual answer - Ollama's chat
    template opens that block as part of the prompt scaffold, so the
    opening tag itself is often never present in `response`, only the
    model's own reasoning text followed by its closing </think>. Left
    unstripped, that reasoning text gets treated as the real answer itself
    (a cleaned prompt, or - for rate_prompts.py - a rating), and can also
    derail delimiter/rating-token detection downstream, e.g. by mentioning
    a valid rating word while merely musing over candidates before
    reaching its actual conclusion. Strips a properly paired
    <think>...</think> block if present; otherwise, if only a closing tag
    showed up (the implicit-opening-tag case), discards everything up to
    and including it."""
    without_paired = re.sub(r"<think>.*?</think>\s*", "", raw_response, flags=re.DOTALL)
    if without_paired != raw_response:
        return without_paired
    if "</think>" in raw_response:
        return raw_response.rsplit("</think>", 1)[1].lstrip()
    return raw_response


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
