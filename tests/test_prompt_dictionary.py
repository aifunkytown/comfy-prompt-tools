"""
Builds PROMPT_DICTIONARY.md - a living, always-regenerated snapshot of the
exact system/user prompts this app currently sends to Ollama, for every
supported cleaning model-config x style combination and a representative
sample of Variations aspects, without ever calling an LLM (see
_no_network_calls). Run it as part of a regression pass, or standalone, any
time you want to see what the app is actually asking for without running it
yourself - a real prompt-wording change shows up as a real diff to the
committed .md file; a no-op refactor doesn't.

Only ever reflects the checked-in base prompt/style text - a personal
*.local.json addendum (system_prompt_addendum, style_adds) is deliberately
excluded (see _no_local_addenda), so the output is identical no matter whose
machine generated it, and never leaks a contributor's personal (often
explicit) customization into a file meant to be committed and shared.

Usage:
    python tests/test_prompt_dictionary.py
"""

import random
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from comfy_prompt_tools import clean_prompts, generate_prompt_variations  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "PROMPT_DICTIONARY.md"

CLEAN_PROMPT_CONFIGS = {
    "Gemma": clean_prompts.SYSTEM_PROMPT_CONFIG_PATH,
    "Qwen": clean_prompts.SYSTEM_PROMPT_CONFIG_PATH.parent / "clean_prompts_qwen.json",
}
STYLE_CONFIGS = {
    "Realistic": clean_prompts.DEFAULT_STYLE_CONFIG_PATH,
    "Anime": clean_prompts.DEFAULT_STYLE_CONFIG_PATH.parent / "style_anime.json",
    "Oil Painting": clean_prompts.DEFAULT_STYLE_CONFIG_PATH.parent / "style_oil_painting.json",
}
VARIATION_ASPECTS = ["hair color", "breast size", "body shape"]
VARIATION_EXAMPLE_PROMPT = (
    "a 24-year-old woman with long wavy hair, standing in a sunlit meadow, "
    "wearing a flowing summer dress, soft natural lighting"
)
# Fixed seed so the one vocabulary value build_variation_request()'s combo
# picker chooses per aspect stays the same across regenerations - the doc's
# diffs should only ever reflect a real prompt/vocab change, not RNG churn.
VARIATION_RANDOM_SEED = 42


@contextmanager
def _no_network_calls():
    """Guards this whole generator against ever accidentally making a real
    Ollama call - every function used here is only supposed to build prompt
    text, never send it. Turns a future regression that starts doing so
    into a loud failure instead of a silent, slow test."""
    with patch(
        "urllib.request.urlopen",
        side_effect=AssertionError("test_prompt_dictionary must never make a real network/Ollama call"),
    ):
        yield


@contextmanager
def _no_local_addenda():
    """clean_prompts.build_system_prompt()/resolve_style_adds() both merge
    in a *.local.json addendum if present - gitignored, personal (often
    explicit) customization that must never end up baked into this
    checked-in, shared file. Forcing load_local_text() to report "no local
    file" for the duration of this generator keeps the output identical
    regardless of whose machine ran it."""
    with patch.object(clean_prompts, "load_local_text", return_value=""):
        yield


def _fence(text):
    return f"```text\n{text}\n```"


def _build_cleaning_section():
    lines = [
        "## Cleaning prompts (clean_prompts.py)",
        "",
        "Every Prompt-directions file (Gemma = `clean_prompts.json`, Qwen = "
        "`clean_prompts_qwen.json`) crossed with every visual style "
        "(`style_realism.json` / `style_anime.json` / `style_oil_painting.json`). "
        "Gemma's base prompt has no `{STYLE}` placeholder, so style has no "
        "effect on it - included anyway, to make that asymmetry visible "
        "instead of hiding it.",
        "",
    ]
    for model_label, prompt_config in CLEAN_PROMPT_CONFIGS.items():
        for style_label, style_config in STYLE_CONFIGS.items():
            system_prompt, image_prompt = clean_prompts.resolve_system_prompts(prompt_config, style_config)
            lines += [
                f"### {model_label} + {style_label}",
                "",
                "**System prompt (text rewrite):**",
                "",
                _fence(system_prompt),
                "",
                "**System prompt (image description):**",
                "",
                _fence(image_prompt),
                "",
            ]
    return lines


def _build_variations_section():
    vocab, _random_exclude, multi_select, _explicit_aspects = generate_prompt_variations.load_vocab(
        generate_prompt_variations.DEFAULT_VOCAB_PATH
    )
    lines = [
        "## Variations prompts (generate_prompt_variations.py)",
        "",
        "One aspect at a time, for a single (`count=1`) variation of this example "
        "source prompt, letting the same value-picking logic a real run uses choose "
        "one vocabulary value per aspect (seeded for reproducible output across "
        "regenerations):",
        "",
        f"> {VARIATION_EXAMPLE_PROMPT}",
        "",
    ]
    random.seed(VARIATION_RANDOM_SEED)
    for aspect in VARIATION_ASPECTS:
        system_prompt, user_prompt = generate_prompt_variations.build_variation_request(
            VARIATION_EXAMPLE_PROMPT, aspect, 1, vocab, multi_select,
        )
        lines += [
            f"### {aspect}",
            "",
            "**System prompt:**",
            "",
            _fence(system_prompt),
            "",
            "**User prompt:**",
            "",
            _fence(user_prompt),
            "",
        ]
    return lines


def build_prompt_dictionary():
    """Returns the full PROMPT_DICTIONARY.md text. Never touches the network
    (_no_network_calls) and never bakes in a personal *.local.json addendum
    (_no_local_addenda)."""
    with _no_network_calls(), _no_local_addenda():
        header = [
            "# Prompt Dictionary",
            "",
            "Auto-generated by `tests/test_prompt_dictionary.py` - do not hand-edit, "
            "your changes will be overwritten the next time it runs. A living "
            "snapshot of the exact instructions this app currently sends to Ollama "
            "for each supported combination, built without ever calling an LLM, so "
            "you can see what the app is actually asking for without running it "
            "yourself. Regenerate with:",
            "",
            "```bash",
            "python tests/test_prompt_dictionary.py",
            "```",
            "",
        ]
        body = header + _build_cleaning_section() + _build_variations_section()
        return "\n".join(body).rstrip() + "\n"


def test_prompt_dictionary():
    """Regenerates PROMPT_DICTIONARY.md and sanity-checks the result -
    pytest-discoverable (test_ prefix) if this project ever adopts pytest,
    and directly runnable standalone in the meantime (see __main__ below)."""
    content = build_prompt_dictionary()

    assert content.startswith("# Prompt Dictionary")
    for model_label in CLEAN_PROMPT_CONFIGS:
        for style_label in STYLE_CONFIGS:
            assert f"### {model_label} + {style_label}" in content, f"missing {model_label} + {style_label} section"
    for aspect in VARIATION_ASPECTS:
        assert f"### {aspect}" in content, f"missing {aspect} section"

    # A personal .local.json addendum (if this machine has one) must never
    # leak into the committed file - check against whatever it actually
    # contains here, rather than hardcoding its (often explicit) wording.
    for prompt_config in CLEAN_PROMPT_CONFIGS.values():
        addendum = clean_prompts.load_local_text(prompt_config, "system_prompt_addendum")
        assert not addendum or addendum not in content, f"a personal addendum from {prompt_config} leaked into {OUTPUT_PATH.name}"
    for style_config in STYLE_CONFIGS.values():
        style_addendum = clean_prompts.load_local_text(style_config, "style_adds")
        assert not style_addendum or style_addendum not in content, f"a personal style addendum from {style_config} leaked into {OUTPUT_PATH.name}"

    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(content)} characters)")


if __name__ == "__main__":
    test_prompt_dictionary()
    print("PROMPT DICTIONARY TEST PASSED")
