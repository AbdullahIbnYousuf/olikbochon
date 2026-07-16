"""Deterministic text preprocessing for the Version 1 baseline."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

import pandas as pd


PROMPT_MARKER = "__PROMPT__"
CONTEXT_PRESENT_MARKER = "__CONTEXT_PRESENT__"
CONTEXT_MARKER = "__CONTEXT__"
RESPONSE_MARKER = "__RESPONSE__"

_WHITESPACE_RE = re.compile(r"\s+")
_MISSING_CONTEXT_MARKERS = frozenset({"", "[null]", "nan", "none", "null", "na", "n/a"})


def _is_scalar_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def normalize_text(value: Any) -> str:
    """Convert a scalar to NFC text and collapse whitespace without lexical rewriting."""
    if _is_scalar_missing(value):
        return ""
    normalized = unicodedata.normalize("NFC", str(value))
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def normalize_context(value: Any) -> str:
    """Normalize context and map common missing-context sentinels to an empty string."""
    normalized = normalize_text(value)
    if normalized.casefold() in _MISSING_CONTEXT_MARKERS:
        return ""
    return normalized


def context_is_present(value: Any) -> bool:
    """Return whether context remains after deterministic missing-value normalization."""
    return bool(normalize_context(value))


def build_marked_text(prompt: Any, context: Any, response: Any) -> str:
    """Build the exact field-marked input consumed by both TF-IDF branches."""
    prompt_text = normalize_text(prompt)
    context_text = normalize_context(context)
    response_text = normalize_text(response)
    present = int(bool(context_text))
    return (
        f"{PROMPT_MARKER}\n{prompt_text}\n\n"
        f"{CONTEXT_PRESENT_MARKER}\n{present}\n\n"
        f"{CONTEXT_MARKER}\n{context_text}\n\n"
        f"{RESPONSE_MARKER}\n{response_text}"
    )


def build_text_series(frame: pd.DataFrame) -> pd.Series:
    """Construct marked text for a validated frame without mutating it."""
    return pd.Series(
        (
            build_marked_text(row.prompt_bn, row.context, row.response_bn)
            for row in frame[["prompt_bn", "context", "response_bn"]].itertuples(index=False)
        ),
        index=frame.index,
        dtype="string",
        name="model_text",
    )


def context_presence_series(frame: pd.DataFrame) -> pd.Series:
    """Return the normalized context-presence flag for aggregate reporting."""
    return frame["context"].map(context_is_present).astype(bool)
