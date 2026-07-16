import unicodedata

import pandas as pd

from olikbochon.preprocessing import (
    build_marked_text,
    context_is_present,
    normalize_context,
    normalize_text,
)


def test_unicode_nfc_and_whitespace_normalization() -> None:
    decomposed = "Cafe\u0301\t  বাংলা\nশব্দ"
    result = normalize_text(decomposed)
    assert result == "Café বাংলা শব্দ"
    assert unicodedata.is_normalized("NFC", result)


def test_missing_context_markers() -> None:
    for value in (None, pd.NA, float("nan"), "", "   ", "[NULL]", "null", "NaN"):
        assert normalize_context(value) == ""
        assert not context_is_present(value)
    assert normalize_context("  প্রাসঙ্গিক   তথ্য ") == "প্রাসঙ্গিক তথ্য"
    assert context_is_present("প্রাসঙ্গিক তথ্য")


def test_field_markers_and_presence_are_deterministic() -> None:
    first = build_marked_text(" প্রশ্ন ", "[NULL]", " উত্তর ")
    second = build_marked_text(" প্রশ্ন ", "[NULL]", " উত্তর ")
    assert first == second
    assert first == (
        "__PROMPT__\nপ্রশ্ন\n\n"
        "__CONTEXT_PRESENT__\n0\n\n"
        "__CONTEXT__\n\n\n"
        "__RESPONSE__\nউত্তর"
    )
