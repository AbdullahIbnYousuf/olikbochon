"""Self-contained default-only normalizer for the Version 3 Kaggle runtime."""

from __future__ import annotations

import hashlib
import random
import unicodedata
from collections.abc import Callable

from ftfy import fix_text

from . import v3_normalizer_constants as const


NORMALIZER_REFERENCE_REPOSITORY = "csebuetnlp/normalizer"
NORMALIZER_REFERENCE_REVISION = "d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9"
FTFY_VERSION = "6.0.3"
WCWIDTH_VERSION = "0.8.2"
REFERENCE_CORPUS_DIGEST = "49020fd6767f4dc5d5d5a9d60d982753f272a23b777f55f7432338122f91cd58"


class UnsupportedNormalizationConfiguration(ValueError):
    """Raised when code requests a non-audited normalizer option."""


def normalize_default(
    text: str,
    unicode_norm: str = "NFKC",
    punct_replacement: str | None = None,
    url_replacement: str | None = None,
    emoji_replacement: str | None = None,
    apply_unicode_norm_last: bool = True,
) -> str:
    """Reproduce the official normalizer's locked default call only."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    unsupported = {
        "unicode_norm": unicode_norm != "NFKC",
        "punct_replacement": punct_replacement is not None,
        "url_replacement": url_replacement is not None,
        "emoji_replacement": emoji_replacement is not None,
        "apply_unicode_norm_last": apply_unicode_norm_last is not True,
    }
    invalid = [name for name, active in unsupported.items() if active]
    if invalid:
        raise UnsupportedNormalizationConfiguration(
            "Version 3 supports only the audited default normalize(text) configuration; "
            f"unsupported arguments: {', '.join(invalid)}"
        )

    normalized = fix_text(text, normalization="NFC", explain=False)
    normalized = const.SINGLE_QUOTE_REGEX.sub("'", normalized)
    normalized = const.DOUBLE_QUOTE_REGEX.sub('"', normalized)
    normalized = normalized.translate(const.CHAR_REPLACEMENTS)
    normalized = const.UNICODE_REPLACEMENTS_REGEX.sub(
        lambda match: const.UNICODE_REPLACEMENTS.get(
            match.group(0), f"{match.group(1)}\u09cc"
        ),
        normalized,
    )
    normalized = unicodedata.normalize("NFKC", normalized)
    return const.WHITESPACE_HANDLER_REGEX.sub(" ", normalized)


def synthetic_reference_corpus() -> tuple[str, ...]:
    """Return safe deterministic inputs used for local and Kaggle parity gates."""
    fixed = [
        "",
        "বাংলা ভাষা",
        "  বাংলা\t\nভাষা  ",
        "‘বাংলা’ “ভাষা”",
        "ＡＢＣ １２৩",
        "Cafe\u0301 déjà vu",
        "বাংলা 😀 পাঠ",
        "https://example.org/বাংলা?q=১",
        "! বাংলা, English?",
        "âœ” No problems",
        "l’humanitÃ© বাংলা",
        "েশৗ",
        "\u09af\u09bc",
        "শব্দ\u00a0শব্দ\u200bশব্দ",
        ("বাংলা ‘text’ Ａ１２ 😀 \t" * 300),
    ]
    generator = random.Random(42)
    pools = (
        range(0x20, 0x7F),
        range(0x0980, 0x0A00),
        range(0x2000, 0x2070),
        range(0xFF01, 0xFF5F),
        range(0x1F600, 0x1F650),
    )
    randomized: list[str] = []
    for _ in range(64):
        length = generator.randint(0, 80)
        randomized.append(
            "".join(chr(generator.choice(generator.choice(pools))) for _ in range(length))
        )
    return tuple(fixed + randomized)


def corpus_output_digest(normalizer: Callable[[str], str]) -> str:
    """Hash length-delimited outputs without exposing corpus values in logs."""
    digest = hashlib.sha256()
    for value in synthetic_reference_corpus():
        output = normalizer(value).encode("utf-8", errors="surrogatepass")
        digest.update(len(output).to_bytes(8, "big"))
        digest.update(output)
    return digest.hexdigest()


def verify_frozen_reference_corpus() -> dict[str, object]:
    """Fail when the bundled normalizer diverges from authenticated outputs."""
    observed = corpus_output_digest(normalize_default)
    if observed != REFERENCE_CORPUS_DIGEST:
        raise RuntimeError(
            "Bundled normalizer parity failed before data loading: "
            f"expected {REFERENCE_CORPUS_DIGEST}, observed {observed}"
        )
    return {
        "cases": len(synthetic_reference_corpus()),
        "digest": observed,
        "reference_revision": NORMALIZER_REFERENCE_REVISION,
        "ftfy_version": FTFY_VERSION,
        "wcwidth_version": WCWIDTH_VERSION,
    }
