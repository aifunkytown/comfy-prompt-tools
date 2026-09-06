"""
Take one row from a prompt CSV (as produced by extract_image_prompts.py) and
ask a local Ollama model to generate variations of its Positive Prompt that
change one described aspect (e.g. "dress color") while keeping everything
else about the prompt the same. Writes a new CSV with the original prompt as
the first row, followed by the requested number of variations, into a
"Variations" folder next to the input CSV (created if it doesn't exist).

Requires Ollama running locally (ollama serve) with the model already pulled:
    ollama pull gemma4:12b

Usage:
    python generate_prompt_variations.py <csv_path> <row> <aspect> <count>

    python generate_prompt_variations.py prompts.csv 3 "dress color" 5

    # e.g. if row 3's Positive Prompt is "girl in blue dress", this writes a
    # CSV with 6 rows: the original, plus 5 variations each with a different
    # dress color and everything else about the prompt left alone.

    # Multiple aspects can be given at once, separated by "and" or a comma -
    # the model will vary ALL of them together, producing as many distinct
    # combinations as the count allows:
    python generate_prompt_variations.py prompts.csv 3 "hair color and dress color" 10

    # e.g. this writes a CSV with 11 rows: the original, plus 10 variations
    # each with a different hair-color/dress-color combination.

    # Aspects can have a controlled vocabulary of allowed values, so the
    # model picks from your list instead of improvising its own adjectives.
    # Define these once in a vocab JSON file (default: prompt_aspect_vocab.json
    # next to this script) mapping aspect name -> list of values, e.g.:
    #   { "breast size": ["flat", "small", "medium", "large", "huge", "giant", "enormous"] }
    # Any aspect name you pass that matches a key in the vocab file
    # (case-insensitive) will automatically be constrained to that list. Use
    # --vocab to point at a different file.

    # Some aspects can have MULTIPLE values combined in a single variation
    # instead of exactly one (e.g. wearing several accessories at once).
    # Mark these with a reserved "_multi_select" key in the vocab JSON,
    # mapping aspect name -> max number of values to combine:
    #   { "_multi_select": {"accessories": 5}, ... }

    # Instead of naming aspects yourself, you can have N of them chosen at
    # random from the vocab file - useful for exploring combinations you
    # wouldn't have thought to ask for. Omit the aspect argument and pass
    # --random-aspects instead:
    python generate_prompt_variations.py prompts.csv 3 10 --random-aspects 3

    # e.g. this randomly picks 3 aspects from the vocab file (e.g. "hair
    # color", "setting", "pose") and produces 10 variations combining them.

    # To permanently exclude certain aspects from ever being picked by
    # --random-aspects (they're still usable by name), add a reserved
    # "_exclude_from_random" key to the vocab JSON listing their names:
    #   { "_exclude_from_random": ["furry species", "number of subjects"], ... }

    # Aspects that are NSFW/explicit can be tagged with a reserved
    # "_explicit_aspects" key listing their names:
    #   { "_explicit_aspects": ["nudity", "penis size"], ... }
    # These are still usable by name at any time. But for --random-aspects,
    # they're only eligible to be picked for a row if that row's own source
    # prompt explicitly states an age over 18 (e.g. "25-year-old") -
    # otherwise (age unstated, or 18 or under) they're excluded from random
    # selection for that row, the same way --random-aspects excludes
    # anything listed in "_exclude_from_random".

    # "resolution" is a special, structural aspect (see STRUCTURAL_ASPECTS) -
    # rather than describing something to weave into the prompt TEXT, each
    # variation just gets assigned one of the vocab's resolution values
    # directly (same even-coverage cycling as any other single-select vocab
    # aspect), written to a "Resolution" column in the output CSV instead of
    # ever being shown to the model. Requires a non-empty "resolution" list
    # in the vocab file, and is always excluded from --random-aspects (add
    # it to "_exclude_from_random" - already done in the default vocab file)
    # since it's not a narrative/visual choice. rerun_prompts_comfyui.py
    # reads that column and resizes the workflow's Empty Latent Image node
    # accordingly when re-queuing a row that has one set.
    python generate_prompt_variations.py prompts.csv 3 "resolution" 5

    # If combined with other aspects, only the non-resolution ones go to the
    # model for text variation - the same prompt each time is then paired
    # with a different resolution per variation:
    python generate_prompt_variations.py prompts.csv 3 "hair color and resolution" 6
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Prompt text (esp. from booru-tag sources) can contain characters outside
# the Windows console's default codepage (e.g. cp1252) - reconfigure stdout/
# stderr to UTF-8 so printing it never crashes mid-batch.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "gemma4:12b"
DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "prompt_aspect_vocab.json"


def list_ollama_models(url=OLLAMA_URL, timeout=5):
    """Names of models currently pulled in the local Ollama install (via
    /api/tags), sorted alphabetically - empty list if Ollama isn't running
    or unreachable, so callers (e.g. a GUI model dropdown) can fall back to
    DEFAULT_MODEL instead of erroring."""
    base = url.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []
    return sorted(m["name"] for m in data.get("models", []) if m.get("name"))

SEPARATOR = "===VARIATION==="

SYSTEM_PROMPT = (
    "You generate variations of an image-generation prompt. You will be given "
    "an original prompt, one or more aspects of the scene to vary, and a "
    "number of variations to produce.\n\n"
    "Each named aspect describes a CONCEPT, not necessarily an exact word or "
    "phrase already in the prompt - understand what that concept means for "
    "this specific scene and imagine a genuinely different, distinct take on "
    "it for each variation, the way a human artist would if asked to redo the "
    "same scene with that one thing changed. Do not just find-and-replace a "
    "word; if a concept isn't spelled out explicitly in the original prompt "
    "at all, add it in naturally as part of the rewrite.\n\n"
    "If MORE THAN ONE aspect is given, vary ALL of the named aspects "
    "together in every variation, not just one of them at a time. Treat the "
    "task as exploring the combination space across the named aspects: "
    "spread the requested number of variations across as many distinct "
    "combinations of aspect values as you can (e.g. with two aspects "
    "'hair color' and 'dress color' and 10 requested variations, produce up "
    "to 10 different hair-color/dress-color pairings), and do not repeat the "
    "same combination of values twice unless you run out of genuinely "
    "distinct combinations. You MUST keep generating new combinations until "
    "you have produced exactly the requested number of variations - never "
    "stop early.\n\n"
    "Some aspects will come with an explicit allowed list of values in "
    "parentheses right after the aspect name. When a list is given for an "
    "aspect, you MUST pick that aspect's value for each variation from that "
    "exact list only - do not invent new values for it or reword the listed "
    "values. Spread your picks across the full list (covering as much of "
    "the range, including the extremes, as the requested count allows) "
    "instead of clustering on similar or adjacent values, and do not reuse "
    "a value until every other value in the list has been used at least "
    "once. Aspects without a list are still free-form as described above. "
    "EXCEPTION: if the prompt below instead gives you an explicit numbered "
    "list like \"Variation 1: aspect = value\" for every variation, that "
    "list has already handled the non-repetition/spreading for you - just "
    "use the exact value(s) assigned to each variation number, and ignore "
    "the general \"you pick and spread the values\" guidance above for "
    "those aspects.\n\n"
    "A value list marked \"choose 1 to N of these values, combined together\" "
    "means that aspect allows MULTIPLE values at once instead of exactly "
    "one - for each variation, pick a number of items from 1 up to N (vary "
    "how many across the variations too, not always the same count) and "
    "naturally incorporate all of the chosen items together into that one "
    "variation (e.g. a character simultaneously wearing glasses, a choker "
    "necklace, AND a beanie, not three separate variations with one item "
    "each). Spread your combinations across the requested count so "
    "different variations use different items and different counts.\n\n"
    "Some named aspects describe a literal visual element (hair color, "
    "pose, clothing) - for these, change or add that specific element. "
    "Other aspects describe an overall QUALITY or STYLE of the whole image "
    "(e.g. 'absurdity level', 'detail level', 'photography style', 'prompt "
    "length') rather than one visual thing - for these, you must actually "
    "rewrite the prose throughout the prompt so the ENTIRE description "
    "embodies that quality: change word choice, structure, and which "
    "details are included or emphasized everywhere in the text. For "
    "example, a higher absurdity level means the scene and details "
    "themselves must read as genuinely stranger or more surreal, not that "
    "the word 'surreal' gets mentioned; a different detail level means "
    "actually adding or stripping descriptive richness throughout, not "
    "appending the words 'highly detailed'.\n\n"
    "When the named aspect is about race, species, or a fantasy/furry "
    "species (e.g. 'race', 'species/fantasy race', 'furry species'), and "
    "the chosen value is a non-human species (elf, demon, vampire, angel, "
    "robot/android, orc, alien, or any animal species), you must update "
    "every place the subject is referred to so the whole description stays "
    "consistent with that species - do not just insert the species word "
    "while leaving a generic human noun like 'woman', 'man', 'girl', or "
    "'guy' unchanged elsewhere. Replace it with a species-appropriate "
    "phrase such as 'female elf', 'male wolf', or 'female robot' (gender + "
    "species), and remove any now-contradictory human ethnicity wording "
    "(e.g. don't describe someone as both 'Caucasian' and an 'elf'). For "
    "human ethnicities (Caucasian, Black, East Asian, etc.) this doesn't "
    "apply - 'woman'/'man' stays as-is, only the ethnicity word changes.\n\n"
    "NEVER produce a variation that is just the original prompt with the "
    "aspect's value tacked on as an extra trailing word or clause (e.g. "
    "\"...for professional portrait depth, surreal.\" is a FAILURE - it "
    "does not count as a real variation, even if the value technically "
    "appears in the text). Every variation must be a genuine rewrite where "
    "the requested change is woven naturally into the existing sentence(s), "
    "the way a human editor would revise the passage, not appended to it.\n\n"
    "For each variation, rewrite the ENTIRE prompt, keeping everything else "
    "about the scene the same - subject, pose, style, lighting, quality tags, "
    "LoRA tags, etc - except for the named aspect(s). Write out the full "
    "prompt text each time, in the same style and level of detail as the "
    "original; do not describe the change, write the actual rewritten "
    "prompt.\n\n"
    f"Respond with EXACTLY the requested number of variations and nothing "
    f"else - no numbering, no explanation, no preamble or closing remarks. "
    f"Put a line containing only {SEPARATOR} immediately before each "
    f"variation (including the first one)."
)


def hash_prompt(positive, negative):
    combined = (positive.strip() + "\x00" + negative.strip()).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def parse_row_range(s):
    """Parse a row argument: either a single 1-indexed row number ("100")
    or an inclusive range in "100-105" format. Returns a list of row
    numbers."""
    s = s.strip()
    if "-" in s:
        parts = s.split("-")
        if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
            raise argparse.ArgumentTypeError(f"Invalid row range '{s}' - expected format like '100-105'")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            raise argparse.ArgumentTypeError(f"Invalid row range '{s}' - start ({start}) must be <= end ({end})")
        return list(range(start, end + 1))
    if not s.isdigit():
        raise argparse.ArgumentTypeError(f"Invalid row '{s}' - expected a number or a range like '100-105'")
    return [int(s)]


_AGE_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_AGE_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_AGE_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_AGE_NUMERAL_RE = re.compile(r"\b(\d{1,3})[\s-]*(?:years?[\s-]*old|y\.?o\.?)\b", re.IGNORECASE)
_AGE_OF_RE = re.compile(r"\bage[d]?\s*(?:of\s*)?(\d{1,3})\b", re.IGNORECASE)
# Only match actual number words (not e.g. the leading article in "a
# twenty-year-old"), by building the alternation from the known word lists
# rather than accepting any word.
_NUMBER_WORD_ALT = "|".join(sorted(set(_AGE_TENS) | set(_AGE_TEENS) | set(_AGE_ONES), key=len, reverse=True))
_AGE_WORD_RE = re.compile(
    rf"\b((?:{_NUMBER_WORD_ALT})(?:[\s-](?:{_NUMBER_WORD_ALT}))?)[\s-]*years?[\s-]*old\b",
    re.IGNORECASE,
)


def _word_to_age(phrase):
    words = phrase.lower().replace("-", " ").split()
    if len(words) == 1:
        w = words[0]
        return _AGE_TEENS.get(w) or _AGE_TENS.get(w) or _AGE_ONES.get(w)
    if len(words) == 2 and words[0] in _AGE_TENS and words[1] in _AGE_ONES:
        return _AGE_TENS[words[0]] + _AGE_ONES[words[1]]
    return None


def detect_age(text):
    """Best-effort scan of prompt text for an explicitly stated age, either
    numeral ("20-year-old", "aged 35") or spelled out ("twenty-year-old").
    Returns the age as an int, or None if no age is clearly stated."""
    if not text:
        return None
    m = _AGE_NUMERAL_RE.search(text)
    if m:
        return int(m.group(1))
    m = _AGE_OF_RE.search(text)
    if m:
        return int(m.group(1))
    m = _AGE_WORD_RE.search(text)
    if m:
        age = _word_to_age(m.group(1))
        if age is not None:
            return age
    return None


def parse_aspects(aspect_arg):
    """Split an aspect argument like "hair color and dress color" or
    "hair color, dress color" into a list of individual aspects. A plain
    single aspect (no comma/"and") is returned as a one-item list."""
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", aspect_arg.strip(), flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def load_vocab(vocab_path):
    """Load the aspect -> allowed-values map from a JSON file. Missing file
    is not an error (just means no aspect has a controlled vocabulary);
    keys are matched case-insensitively against aspect names.

    Returns (vocab, random_exclude, multi_select, explicit_aspects):
    - random_exclude: set of aspect names (lowercased) pulled from the
      reserved "_exclude_from_random" key, if present - these are still
      usable as normal named aspects, just never picked by --random-aspects.
    - multi_select: {aspect name (lowercased): max count} pulled from the
      reserved "_multi_select" key, if present - these aspects allow
      combining more than one value together in a single variation (e.g.
      wearing several accessories at once) instead of picking exactly one.
    - explicit_aspects: set of aspect names (lowercased) pulled from the
      reserved "_explicit_aspects" key, if present - these are still usable
      by name at any time, but --random-aspects only makes them eligible
      for a row if that row's own source prompt explicitly states an age
      over 18; otherwise they're excluded from random selection for that
      row (see main())."""
    path = Path(vocab_path)
    if not path.is_file():
        return {}, set(), {}, set()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    exclude_raw = raw.pop("_exclude_from_random", [])
    random_exclude = {str(x).strip().lower() for x in exclude_raw}
    multi_select_raw = raw.pop("_multi_select", {})
    multi_select = {str(k).strip().lower(): int(v) for k, v in multi_select_raw.items()}
    explicit_raw = raw.pop("_explicit_aspects", [])
    explicit_aspects = {str(x).strip().lower() for x in explicit_raw}
    vocab = {str(k).strip().lower(): list(v) for k, v in raw.items()}
    return vocab, random_exclude, multi_select, explicit_aspects


def _format_aspect(aspect, vocab, multi_select=None):
    multi_select = multi_select or {}
    values = vocab.get(aspect.lower())
    if not values:
        return aspect
    max_count = multi_select.get(aspect.lower())
    if max_count:
        return f"{aspect} (choose 1 to {max_count} of these values, combined together in the same variation: {', '.join(values)})"
    return f"{aspect} (allowed values only: {', '.join(values)})"


STRUCTURAL_ASPECTS = {"resolution"}
# Aspects that don't describe something to weave into the prompt TEXT -
# instead their per-variation value drives a structural change to the
# ComfyUI workflow itself. Handled entirely outside generate_variations()/
# the LLM: see run_batch()'s resolution_requested handling and
# rerun_prompts_comfyui.py's use of the output CSV's "Resolution" column.


def _cycle_values(values, count):
    """`count` values cycling through a shuffled copy of `values` (Latin-
    square style, same technique _build_combo_sequence uses for a single-
    select vocab aspect) - guarantees values are used as evenly as possible
    rather than picked independently at random."""
    shuffled = list(values)
    random.shuffle(shuffled)
    return [shuffled[n % len(shuffled)] for n in range(count)]


def _build_combo_sequence(aspects, vocab, multi_select, count):
    """Pre-assign exactly which vocab-controlled value(s) each of the
    `count` variations must use, so non-repetition is guaranteed in code
    instead of hoping the model self-polices it. Returns a list of `count`
    dicts {aspect_name: value_or_tuple}, one per variation, covering
    vocab-backed aspects only (freeform aspects aren't included - the model
    still improvises those). Returns None if no named aspect has a vocab.

    For single-select-only aspects, each aspect gets its own independently
    shuffled value list, cycled by its own domain size (variation n uses
    position n % len(that aspect's values)) - a Latin-square-style
    traversal. This guarantees every aspect's values are used as evenly as
    possible (no aspect gets skewed toward a handful of values just because
    a *different* named aspect happens to have a larger value list - purely
    random independent picks per aspect, checked only for joint-pair
    uniqueness, let this happen: e.g. combining a 16-value and a 7-value
    aspect randomly can easily produce one 7-value option 4x while another
    is never picked, even though no exact pair repeats). Joint combos won't
    repeat until at least lcm(domain sizes) variations have been produced
    (equal to the full combo space when domain sizes are coprime), at which
    point they naturally cycle back via the modulo indexing.

    If a multi-select aspect is involved, the combo space (every possible
    subset) is usually enormous, so exact cycling isn't practical - repeats
    there are avoided via rejection sampling instead, same as before."""
    vocab_aspects = [a for a in aspects if a.lower() in vocab and vocab[a.lower()]]
    if not vocab_aspects:
        return None

    all_single_select = all(not multi_select.get(a.lower()) for a in vocab_aspects)

    if all_single_select:
        cycles = {}
        for a in vocab_aspects:
            values = list(vocab[a.lower()])
            random.shuffle(values)
            cycles[a] = values
        return [
            {a: cycles[a][n % len(cycles[a])] for a in vocab_aspects}
            for n in range(count)
        ]

    def sample_one_pick(a):
        values = vocab[a.lower()]
        max_count = multi_select.get(a.lower())
        if max_count:
            k = random.randint(1, min(max_count, len(values)))
            return tuple(sorted(random.sample(values, k)))
        return random.choice(values)

    seen = set()
    combos = []
    attempts_cap = 300
    for _ in range(count):
        for _ in range(attempts_cap):
            combo = tuple((a, sample_one_pick(a)) for a in vocab_aspects)
            if combo not in seen:
                seen.add(combo)
                combos.append(dict(combo))
                break
        else:
            # couldn't find an unused combo within the attempt cap (huge
            # multi-select space is thinning out) - cycle and reuse
            seen.clear()
            combo = tuple((a, sample_one_pick(a)) for a in vocab_aspects)
            seen.add(combo)
            combos.append(dict(combo))
    return combos


def build_variation_request(original_prompt, aspect, count, vocab=None, multi_select=None):
    """Builds the (system_prompt, user_prompt) pair generate_variations()
    sends to Ollama - split out from it so a caller that just wants to see
    (or document - see tests/test_prompt_dictionary.py) what would actually
    be asked doesn't need to duplicate this formatting logic or make a real
    Ollama call to get it."""
    vocab = vocab or {}
    multi_select = multi_select or {}
    aspects = parse_aspects(aspect)

    combos = _build_combo_sequence(aspects, vocab, multi_select, count)

    if combos is not None:
        vocab_context = "\n".join(
            f"- {_format_aspect(a, vocab, multi_select)}"
            for a in aspects if a.lower() in vocab and vocab[a.lower()]
        )
        combo_lines = []
        for i, combo in enumerate(combos, 1):
            parts = []
            for a in aspects:
                if a in combo:
                    value = combo[a]
                    value = ", ".join(value) if isinstance(value, tuple) else value
                    parts.append(f"{a} = {value}")
                else:
                    parts.append(f"{a} = (freeform - your own choice, distinct from other variations)")
            combo_lines.append(f"Variation {i}: {'; '.join(parts)}")
        aspect_section = (
            f"Aspects to vary: {', '.join(aspects)}\n"
            f"{vocab_context}\n\n"
            f"Each variation below has been pre-assigned the EXACT value(s) "
            f"to use for the vocab-controlled aspect(s) - you MUST use "
            f"exactly that value (or exact set of values, for multi-select "
            f"aspects) when writing that variation, woven naturally into "
            f"the prose, not appended as a label. For any aspect marked "
            f"'(freeform - your own choice...)', invent your own distinct "
            f"value as usual - just don't repeat a freeform value you've "
            f"already used in an earlier variation of this batch.\n\n"
            f"{chr(10).join(combo_lines)}"
        )
    elif len(aspects) > 1:
        aspect_lines = "\n".join(f"- {_format_aspect(a, vocab, multi_select)}" for a in aspects)
        aspect_section = (
            f"Aspects to vary (vary ALL of these together in every variation - "
            f"generate as many distinct combinations of them as needed to reach "
            f"the requested count):\n{aspect_lines}"
        )
    else:
        aspect_section = f"Aspect to vary: {_format_aspect(aspects[0], vocab, multi_select)}"

    user_prompt = (
        f"Original prompt:\n{original_prompt}\n\n"
        f"{aspect_section}\n"
        f"Number of variations: {count}"
    )
    return SYSTEM_PROMPT, user_prompt


def generate_variations(original_prompt, aspect, count, model, vocab=None, multi_select=None):
    system_prompt, user_prompt = build_variation_request(original_prompt, aspect, count, vocab, multi_select)
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
        # Some models (e.g. gemma4) support a hidden "thinking" pass before
        # the visible answer. Left on, they can burn the entire num_predict
        # budget on invisible reasoning and return a completely empty
        # response for larger/more complex requests. We only want the
        # rewritten prompts, so disable it.
        "think": False,
        # -1 = unlimited generation (runs until the model stops on its own
        # or hits its context window). This is a local model with no
        # per-token cost, so there's no reason to cap output length and
        # risk truncating larger variation counts/longer prompts.
        "options": {"num_predict": -1},
    }
    req_body = json.dumps(payload).encode("utf-8")

    # Ollama occasionally returns a completely empty response (observed under
    # GPU/VRAM contention from other loaded models) with no error - retry
    # once before giving up.
    attempts = 2
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            OLLAMA_URL,
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        response_text = result.get("response", "")
        if response_text.strip():
            return _extract_variations_list(response_text)

        if attempt < attempts:
            print("Warning: Ollama returned an empty response, retrying...", file=sys.stderr)

    raise ValueError("Ollama returned an empty response (after retry)")


def _extract_variations_list(response_text):
    """Split the model's plain-text response on the SEPARATOR line into
    individual variation strings. Plain text with a distinctive separator is
    far more reliable for small/quantized local models than asking for a
    precise JSON schema, which they frequently don't follow exactly."""
    parts = re.split(rf"(?m)^\s*{re.escape(SEPARATOR)}\s*$", response_text)
    variations = [p.strip() for p in parts if p.strip()]
    if not variations:
        raise ValueError(f"Could not find any variations in the model's response: {response_text[:300]}")
    return variations


def run_batch(csv_path, row_numbers, aspect=None, count=1, output=None, model=DEFAULT_MODEL, vocab_path=None, random_aspects=None, prompt_overrides=None, aspect_values=None):
    """Core logic shared by the CLI (main()) and anything else driving this
    programmatically (e.g. the GUI's Variations tab, via run() below).
    csv_path is a Path, row_numbers a list of 1-indexed ints - not required
    to be contiguous, so a caller can skip rows the user removed from a
    preview list (see parse_row_range for the CLI's contiguous-range
    parsing). aspect/random_aspects mutually exclusive same as the CLI.
    prompt_overrides is an optional {row_num: text} map - when a row_num is
    present, its text replaces that row's own Positive/Cleaned Prompt as the
    original prompt to vary, while every other column (File Name, Negative
    Prompt, etc.) still comes from the CSV row as usual. aspect_values is an
    optional {aspect name (any case): [subset of values]} map - every vocab-
    controlled aspect only ever reads its allowed values back out of the
    `vocab` dict built here (never straight from the file again), so
    narrowing an entry in it before anything else runs is enough to
    restrict that aspect to just those values for this run, without
    touching the vocab file itself. Only narrows an aspect the vocab file
    already defines - it's not a way to invent a new one (silently
    ignored if the name doesn't match an existing vocab key). Raises
    SystemExit on unrecoverable errors, same convention as run_test.py/
    lora_test.py."""
    vocab_path = vocab_path or DEFAULT_VOCAB_PATH
    prompt_overrides = prompt_overrides or {}
    vocab, random_exclude, multi_select, explicit_aspects = load_vocab(vocab_path)
    for name, values in (aspect_values or {}).items():
        key = str(name).strip().lower()
        if key in vocab:
            vocab[key] = list(values)

    if random_aspects is not None and aspect:
        print("Error: provide either an aspect or random_aspects, not both", file=sys.stderr)
        sys.exit(1)
    if random_aspects is None and not aspect:
        print("Error: must provide either an aspect or random_aspects", file=sys.stderr)
        sys.exit(1)

    if random_aspects is not None:
        if random_aspects < 1:
            print("Error: random_aspects must be at least 1", file=sys.stderr)
            sys.exit(1)
        base_eligible = [k for k, v in vocab.items() if v and k not in random_exclude]
        if random_aspects > len(base_eligible):
            print(f"Error: random_aspects {random_aspects} exceeds the {len(base_eligible)} eligible aspect(s) in the vocab file", file=sys.stderr)
            sys.exit(1)

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    bad_rows = [r for r in row_numbers if r < 1 or r > len(rows)]
    if bad_rows:
        print(f"Error: row(s) {', '.join(map(str, bad_rows))} out of range (CSV has {len(rows)} data row(s))", file=sys.stderr)
        sys.exit(1)

    if output and len(row_numbers) > 1:
        print("Error: output path can't be used with a multi-row range (each row needs its own output file) - omit it to use the default 'Variations' folder naming", file=sys.stderr)
        sys.exit(1)

    cleaned_col = next((c for c in fieldnames if "cleaned" in c.lower()), None)

    # For a fixed aspect (not --random-aspects) it's the same for every row
    # in the batch, so resolve it once. For --random-aspects, selection has
    # to happen per row instead (see inside the loop below) since
    # eligibility depends on whether that row's own prompt states an age
    # over 18 - explicit aspects are excluded from random selection unless
    # it does.
    if random_aspects is None:
        aspects_list = parse_aspects(aspect)
        aspects_changed = ", ".join(aspects_list)
        matched_vocab = [a for a in aspects_list if a.lower() in vocab]
        for a in matched_vocab:
            max_count = multi_select.get(a.lower())
            tags = []
            if max_count:
                tags.append(f"up to {max_count} combined per variation")
            if a.lower() in explicit_aspects:
                tags.append("explicit")
            suffix = f" ({', '.join(tags)})" if tags else ""
            print(f"Using controlled vocabulary for '{a}'{suffix}: {', '.join(vocab[a.lower()])}")

    succeeded = 0
    failed = 0
    for idx, row_num in enumerate(row_numbers, 1):
        if len(row_numbers) > 1:
            print(f"\n=== Row {row_num} ({idx}/{len(row_numbers)}) ===")

        try:
            source_row = rows[row_num - 1]
            cleaned_text = (source_row.get(cleaned_col) or "").strip() if cleaned_col else ""
            original_prompt = prompt_overrides.get(row_num) or cleaned_text or (source_row.get("Positive Prompt") or "").strip()
            if not original_prompt:
                print(f"Error: row {row_num} has no Positive Prompt (or Cleaned Prompt) text - skipping", file=sys.stderr)
                failed += 1
                continue
            if cleaned_text:
                print(f"Using '{cleaned_col}' column as the source prompt")

            if random_aspects is not None:
                age = detect_age(original_prompt)
                row_eligible = [k for k, v in vocab.items() if v and k not in random_exclude]
                if age is None or age <= 18:
                    row_eligible = [a for a in row_eligible if a not in explicit_aspects]
                if random_aspects > len(row_eligible):
                    age_note = "no age over 18 detected in this row's prompt, so explicit aspects are excluded" if (age is None or age <= 18) else "explicit aspects are eligible"
                    print(f"Error: random_aspects {random_aspects} exceeds the {len(row_eligible)} eligible aspect(s) for row {row_num} ({age_note}) - skipping", file=sys.stderr)
                    failed += 1
                    continue
                chosen = random.sample(row_eligible, random_aspects)
                row_aspect = " and ".join(chosen)
                age_desc = f"age {age} detected" if age is not None else "no age detected"
                print(f"Randomly selected {random_aspects} aspect(s) for row {row_num} ({age_desc}): {', '.join(chosen)}")

                aspects_list = parse_aspects(row_aspect)
                aspects_changed = ", ".join(aspects_list)
                matched_vocab = [a for a in aspects_list if a.lower() in vocab]
                for a in matched_vocab:
                    max_count = multi_select.get(a.lower())
                    tags = []
                    if max_count:
                        tags.append(f"up to {max_count} combined per variation")
                    if a.lower() in explicit_aspects:
                        tags.append("explicit")
                    suffix = f" ({', '.join(tags)})" if tags else ""
                    print(f"Using controlled vocabulary for '{a}'{suffix}: {', '.join(vocab[a.lower()])}")
            else:
                row_aspect = aspect

            row_aspects_list = parse_aspects(row_aspect)
            resolution_requested = any(a.lower() == "resolution" for a in row_aspects_list)
            if resolution_requested and not vocab.get("resolution"):
                print("Error: 'resolution' aspect requires a non-empty 'resolution' list in the vocab file", file=sys.stderr)
                sys.exit(1)
            text_aspect = ", ".join(a for a in row_aspects_list if a.lower() != "resolution")

            print(f"Original prompt (row {row_num}): {original_prompt[:100]}")
            print(f"Generating {count} variation(s) of '{row_aspect}' via {model} ...")

            try:
                if text_aspect:
                    variations = generate_variations(original_prompt, text_aspect, count, model, vocab, multi_select)
                else:
                    # resolution was the only requested aspect - nothing in
                    # the prompt text itself changes, so skip the LLM call
                    # entirely and just repeat the original prompt.
                    variations = [original_prompt] * count
            except Exception as e:
                print(f"Error generating variations for row {row_num}: {e}", file=sys.stderr)
                failed += 1
                continue

            resolutions = _cycle_values(vocab["resolution"], count) if resolution_requested else None

            if len(variations) < count:
                print(f"Warning: model returned {len(variations)} variation(s), expected {count}", file=sys.stderr)
            elif len(variations) > count:
                variations = variations[:count]

            negative = (source_row.get("Negative Prompt") or "").strip()
            other = (source_row.get("Other Parameters") or "").strip()
            source_format = source_row.get("Source Format") or ""
            base_name = Path(source_row.get("File Name") or f"row{row_num}").stem

            if output:
                output_path = Path(output)
            else:
                variations_dir = csv_path.parent / "Variations"
                variations_dir.mkdir(parents=True, exist_ok=True)
                output_path = variations_dir / f"{csv_path.stem}_row{row_num}_variations.csv"

            out_fieldnames = ["Variation", "File Name", "Positive Prompt", "Aspects Changed", "Negative Prompt", "Other Parameters", "Source Format", "Prompt Hash (SHA-256)"]
            if resolution_requested:
                out_fieldnames.append("Resolution")
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=out_fieldnames)
                writer.writeheader()

                original_row = {
                    "Variation": "original",
                    "File Name": base_name,
                    "Positive Prompt": original_prompt,
                    "Aspects Changed": "",
                    "Negative Prompt": negative,
                    "Other Parameters": other,
                    "Source Format": source_format,
                    "Prompt Hash (SHA-256)": hash_prompt(original_prompt, negative),
                }
                if resolution_requested:
                    # Unvaried, same as every other aspect on the original row -
                    # rerun_prompts_comfyui.py leaves the workflow's own
                    # Empty Latent Image size alone when this is blank.
                    original_row["Resolution"] = ""
                writer.writerow(original_row)

                for i, variation_text in enumerate(variations, 1):
                    variation_row = {
                        "Variation": i,
                        "File Name": f"{base_name}_var{i}",
                        "Positive Prompt": variation_text,
                        "Aspects Changed": aspects_changed,
                        "Negative Prompt": negative,
                        "Other Parameters": other,
                        "Source Format": source_format,
                        "Prompt Hash (SHA-256)": hash_prompt(variation_text, negative),
                    }
                    if resolution_requested:
                        variation_row["Resolution"] = resolutions[i - 1]
                    writer.writerow(variation_row)

            print(f"Wrote {1 + len(variations)} row(s) to {output_path}")
            succeeded += 1
        except Exception as e:
            print(f"Error processing row {row_num}: {e}", file=sys.stderr)
            failed += 1
            continue

    if len(row_numbers) > 1:
        print(f"\nDone. {succeeded} row(s) succeeded, {failed} failed.")
        if failed and not succeeded:
            sys.exit(1)


def run(config_path):
    """JSON-config-driven entry point, same convention as
    run_test.run()/lora_test.run() (a caller like the GUI can drive this
    without going through argparse). Config file format:
        {
            "csv_path": "...",
            "row": "100" or "100-105",                  // or "rows" instead
            "rows": [100, 102, 105],                     // explicit, not required to be contiguous
            "aspect": "hair color and dress color",   // or "random_aspects" instead
            "random_aspects": 3,
            "count": 10,
            "model": "gemma4:12b",                      // optional
            "vocab_path": "...",                        // optional
            "output": "...",                            // optional, single-row only
            "prompt_overrides": {"100": "a new prompt for row 100"},  // optional
            "aspect_values": {"hair color": ["red", "blue"]}  // optional - narrows a
                                                              //   vocab-controlled aspect
                                                              //   to just these values
                                                              //   for this run
        }
    "rows" takes precedence over "row" if both are present.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "rows" in config:
        row_numbers = [int(r) for r in config["rows"]]
    else:
        row_numbers = parse_row_range(str(config["row"]))
    run_batch(
        csv_path=Path(config["csv_path"]),
        row_numbers=row_numbers,
        aspect=config.get("aspect"),
        count=config["count"],
        output=config.get("output"),
        model=config.get("model", DEFAULT_MODEL),
        vocab_path=config.get("vocab_path") or DEFAULT_VOCAB_PATH,
        random_aspects=config.get("random_aspects"),
        prompt_overrides={int(k): v for k, v in config.get("prompt_overrides", {}).items()},
        aspect_values=config.get("aspect_values"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="CSV file to read the source row from")
    parser.add_argument("row", type=parse_row_range, help="1-indexed row number (matching the CSV's data rows) to base variations on, e.g. '100'. Also accepts an inclusive range like '100-105' to process multiple rows in one run, each getting its own output CSV")
    parser.add_argument("aspect", nargs="?", default=None, help="The aspect of the prompt to vary, e.g. 'dress color'. Multiple aspects can be given at once, joined with 'and' or a comma, e.g. 'hair color and dress color' - the model will vary all of them together across the requested count. Omit this and use --random-aspects instead to have aspects chosen randomly from the vocab file")
    parser.add_argument("count", type=int, help="Number of variations to generate")
    parser.add_argument("--output", default=None, help="Output CSV path (default: <csv name>_row<row>_variations.csv, in a 'Variations' folder next to the input CSV)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help=f"JSON file mapping aspect name -> list of allowed values (default: {DEFAULT_VOCAB_PATH.name} next to this script, if present)")
    parser.add_argument("--random-aspects", type=int, default=None, metavar="N", help="Instead of specifying 'aspect', randomly choose N aspects from the vocab file (only aspects with a non-empty value list are eligible)")
    args = parser.parse_args()

    run_batch(
        csv_path=Path(args.csv_path),
        row_numbers=args.row,
        aspect=args.aspect,
        count=args.count,
        output=args.output,
        model=args.model,
        vocab_path=args.vocab,
        random_aspects=args.random_aspects,
    )


if __name__ == "__main__":
    main()
