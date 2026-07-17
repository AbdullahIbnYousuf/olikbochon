from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from olikbochon.bangla_normalizer import normalize as reference_normalize
from olikbochon.v3_default_normalizer import (
    REFERENCE_CORPUS_DIGEST,
    UnsupportedNormalizationConfiguration,
    corpus_output_digest,
    normalize_default,
    synthetic_reference_corpus,
    verify_frozen_reference_corpus,
)


ROOT = Path(__file__).resolve().parents[1]


def test_default_only_normalizer_matches_authenticated_reference_corpus() -> None:
    for value in synthetic_reference_corpus():
        assert normalize_default(value) == reference_normalize(value)
    assert corpus_output_digest(reference_normalize) == REFERENCE_CORPUS_DIGEST
    assert verify_frozen_reference_corpus()["cases"] == len(synthetic_reference_corpus())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"punct_replacement": " "},
        {"url_replacement": "URL"},
        {"emoji_replacement": "EMOJI"},
        {"unicode_norm": "NFC"},
        {"apply_unicode_norm_last": False},
    ],
)
def test_default_only_normalizer_rejects_unsupported_arguments(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(UnsupportedNormalizationConfiguration, match="audited default"):
        normalize_default("বাংলা", **kwargs)  # type: ignore[arg-type]


def test_default_path_preserves_url_punctuation_and_emoji() -> None:
    value = "https://example.org/বাংলা?q=১ !? 😀"
    result = normalize_default(value)
    assert "https://example.org/বাংলা?q=১" in result
    assert "!?" in result
    assert "😀" in result


def test_startup_parity_gate_detects_corruption(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("olikbochon.v3_default_normalizer")
    monkeypatch.setattr(module, "REFERENCE_CORPUS_DIGEST", "0" * 64)
    with pytest.raises(RuntimeError, match="parity failed"):
        module.verify_frozen_reference_corpus()


def test_runtime_normalizer_source_has_no_regex_or_emoji_import() -> None:
    sources = [
        ROOT / "src" / "olikbochon" / "v3_default_normalizer.py",
        ROOT / "src" / "olikbochon" / "v3_normalizer_constants.py",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "import regex" not in joined
    assert "import emoji" not in joined
    assert "\\p{punct}" not in joined


def test_bundled_ftfy_and_wcwidth_override_ambient_packages() -> None:
    import ftfy
    import wcwidth

    assert Path(ftfy.__file__).resolve().is_relative_to(ROOT / "src")
    assert Path(wcwidth.__file__).resolve().is_relative_to(ROOT / "src")
    assert ftfy.__version__ == "6.0.3"
    assert wcwidth.__version__ == "0.8.2"
