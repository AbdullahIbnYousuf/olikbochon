"""Deterministic lexical normalization shared by V5 audit, training, and inference."""

from __future__ import annotations

import unicodedata
from typing import Any


ZERO_WIDTH_CODEPOINTS = frozenset(
    {
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # byte-order mark / zero-width no-break space
    }
)
NUMERAL_TRANSLATION = str.maketrans(
    "০১২৩৪৫৬৭৮৯٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "012345678901234567890123456789",
)
PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "।": ".",
        "॥": ".",
        "…": "...",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "‹": "'",
        "›": "'",
        "؟": "?",
        "，": ",",
        "．": ".",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
    }
)


def _lower_latin_only(text: str) -> str:
    lowered: list[str] = []
    for character in text:
        name = unicodedata.name(character, "")
        lowered.append(character.lower() if "LATIN" in name else character)
    return "".join(lowered)


def normalize_v5(value: Any) -> str:
    """Normalize lexical equivalences without deleting words, entities, or numerals."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    text = "".join(character for character in text if character not in ZERO_WIDTH_CODEPOINTS)
    text = text.translate(NUMERAL_TRANSLATION).translate(PUNCTUATION_TRANSLATION)
    text = _lower_latin_only(text)
    return " ".join(text.split())


def strip_v5_punctuation(value: Any) -> str:
    """Replace Unicode punctuation with spaces after applying the frozen normalization."""
    normalized = normalize_v5(value)
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def tokenize_v5(value: Any) -> tuple[str, ...]:
    """Tokenize letters and numerals with no learned vocabulary or external model."""
    text = normalize_v5(value)
    tokens: list[str] = []
    current: list[str] = []
    for character in text:
        if unicodedata.category(character)[0] in {"L", "M", "N"}:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)
