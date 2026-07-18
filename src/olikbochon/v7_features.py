"""Frozen aggregate evidence features for V7 retrieval candidates M, N, and P."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Sequence

import numpy as np

from .v5_normalization import normalize_v5, tokenize_v5
from .v7_retrieval import RetrievedPassage


V4A_FEATURES = (
    "ctx_present",
    "word_overlap_ratio",
    "numbers_supported",
    "number_overlap_ratio",
    "has_numbers_in_response",
    "is_substring",
    "response_len_chars",
    "context_len_chars",
)

TOP5_FEATURES = (
    "maximum_response_token_coverage",
    "maximum_response_substring_support",
    "maximum_numeric_consistency",
    "maximum_character_similarity",
    "mean_top3_retrieval_score",
    "rank1_rank2_score_gap",
    "accepted_passage_count",
)


def _coverage(response_tokens: Sequence[str], evidence_tokens: Sequence[str]) -> float:
    if not response_tokens:
        return 1.0
    overlap = Counter(response_tokens) & Counter(evidence_tokens)
    return float(sum(overlap.values()) / len(response_tokens))


def _numerals(value: Any) -> tuple[str, ...]:
    return tuple(token for token in tokenize_v5(value) if token.isdecimal())


def _numeric_consistency(response: Any, evidence: Any) -> float:
    response_numbers = _numerals(response)
    if not response_numbers:
        return 1.0
    return _coverage(response_numbers, _numerals(evidence))


def accepted_passages(
    passages: Sequence[RetrievedPassage], cutoff: float
) -> tuple[RetrievedPassage, ...]:
    return tuple(passage for passage in passages if passage.score >= cutoff)


def v4a_features(
    response: Any,
    passages: Sequence[RetrievedPassage],
    *,
    cutoff: float,
) -> dict[str, float]:
    """Reproduce the eight explicit evidence features from accepted top-1 evidence."""
    accepted = accepted_passages(passages[:1], cutoff)
    evidence = accepted[0].snippet if accepted else ""
    normalized_response = normalize_v5(response)
    normalized_evidence = normalize_v5(evidence)
    response_tokens = tokenize_v5(normalized_response)
    evidence_tokens = tokenize_v5(normalized_evidence)
    response_numbers = _numerals(normalized_response)
    number_ratio = _numeric_consistency(normalized_response, normalized_evidence)
    return {
        "ctx_present": float(bool(accepted)),
        "word_overlap_ratio": _coverage(response_tokens, evidence_tokens),
        "numbers_supported": float(not response_numbers or number_ratio == 1.0),
        "number_overlap_ratio": number_ratio,
        "has_numbers_in_response": float(bool(response_numbers)),
        "is_substring": float(
            bool(normalized_response) and normalized_response in normalized_evidence
        ),
        "response_len_chars": float(len(normalized_response)),
        "context_len_chars": float(len(normalized_evidence)),
    }


def top5_features(
    response: Any,
    passages: Sequence[RetrievedPassage],
    *,
    cutoff: float,
) -> dict[str, float]:
    """Aggregate only bounded passage evidence; never concatenate article bodies."""
    bounded = tuple(passages[:5])
    accepted = accepted_passages(bounded, cutoff)
    normalized_response = normalize_v5(response)
    response_tokens = tokenize_v5(normalized_response)
    coverages = [
        _coverage(response_tokens, tokenize_v5(passage.snippet)) for passage in accepted
    ]
    substring = [
        float(bool(normalized_response) and normalized_response in normalize_v5(passage.snippet))
        for passage in accepted
    ]
    numeric = [_numeric_consistency(normalized_response, passage.snippet) for passage in accepted]
    similarities = [
        SequenceMatcher(None, normalized_response, normalize_v5(passage.snippet)).ratio()
        for passage in accepted
    ]
    scores = [float(passage.score) for passage in bounded]
    top3 = scores[:3]
    return {
        "maximum_response_token_coverage": max(coverages, default=0.0),
        "maximum_response_substring_support": max(substring, default=0.0),
        "maximum_numeric_consistency": max(numeric, default=0.0),
        "maximum_character_similarity": max(similarities, default=0.0),
        "mean_top3_retrieval_score": float(np.mean(top3)) if top3 else 0.0,
        "rank1_rank2_score_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "accepted_passage_count": float(len(accepted)),
    }


def feature_vector(record: dict[str, float], names: Sequence[str]) -> np.ndarray:
    values = np.asarray([record[name] for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("V7 evidence features contain NaN or Inf")
    return values
