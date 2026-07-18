from __future__ import annotations

from olikbochon.v5_normalization import normalize_v5, strip_v5_punctuation, tokenize_v5


def test_v5_normalization_is_deterministic_and_collapses_equivalences() -> None:
    source = "  ‘TEST’\u200b—১২৩  \n  ٤৫৬।  "
    expected = "'test'-123 456."
    assert normalize_v5(source) == expected
    assert normalize_v5(source) == normalize_v5(source)
    assert normalize_v5(normalize_v5(source)) == expected


def test_v5_normalization_preserves_bengali_words_entities_and_numerals() -> None:
    assert normalize_v5("বাংলাদেশে OpenAI ২০২৬") == "বাংলাদেশে openai 2026"
    assert tokenize_v5("বাংলাদেশে OpenAI ২০২৬") == ("বাংলাদেশে", "openai", "2026")


def test_punctuation_stripping_uses_the_same_normalization() -> None:
    assert strip_v5_punctuation("“ঢাকা”—২০২৬।") == "ঢাকা 2026"
