"""Deterministic aggregate-only near-duplicate audits for labeled Bengali text."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from olikbochon.preprocessing import build_text_series


SIMILARITY_THRESHOLD = 0.97
LENGTH_RATIO_THRESHOLD = 0.90
BLOCK_SIZE = 256


class NearDuplicateAuditError(RuntimeError):
    """Raised when high-confidence near duplicates invalidate a data role."""


@dataclass(frozen=True)
class NearDuplicateAudit:
    """Aggregate cross-partition similarity facts with anonymous row positions."""

    left_name: str
    right_name: str
    left_rows: int
    right_rows: int
    high_confidence_pairs: int
    affected_left_rows: tuple[int, ...]
    affected_right_rows: tuple[int, ...]
    label_agreement_pairs: int
    label_conflict_pairs: int
    maximum_similarity: float
    row_max_similarity_percentiles: dict[str, float]


def _fit_character_features(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    left_text = build_text_series(left).astype(str).to_numpy(dtype=str)
    right_text = build_text_series(right).astype(str).to_numpy(dtype=str)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        lowercase=False,
        strip_accents=None,
        dtype=np.float32,
        norm="l2",
    )
    combined = vectorizer.fit_transform(np.concatenate([left_text, right_text])).tocsr()
    return (
        combined[: len(left_text)],
        combined[len(left_text) :],
        np.char.str_len(left_text).astype(np.int64),
        np.char.str_len(right_text).astype(np.int64),
    )


def audit_near_duplicates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> NearDuplicateAudit:
    """Audit high-confidence cross-partition pairs in bounded dense blocks."""
    left_matrix, right_matrix, left_lengths, right_lengths = _fit_character_features(
        left, right
    )
    left_labels = left["label"].to_numpy(dtype=np.int64)
    right_labels = right["label"].to_numpy(dtype=np.int64)
    affected_left: set[int] = set()
    affected_right: set[int] = set()
    agreement = 0
    conflicts = 0
    pair_count = 0
    row_maxima = np.zeros(len(left), dtype=np.float32)

    safe_right_lengths = np.maximum(right_lengths, 1)
    for start in range(0, len(left), BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, len(left))
        similarities = cosine_similarity(
            left_matrix[start:stop], right_matrix, dense_output=True
        ).astype(np.float32, copy=False)
        row_maxima[start:stop] = similarities.max(axis=1)
        safe_left = np.maximum(left_lengths[start:stop], 1)[:, None]
        ratios = np.minimum(safe_left, safe_right_lengths[None, :]) / np.maximum(
            safe_left, safe_right_lengths[None, :]
        )
        flagged = np.argwhere(
            (similarities >= SIMILARITY_THRESHOLD) & (ratios >= LENGTH_RATIO_THRESHOLD)
        )
        for local_left, right_index in flagged:
            left_index = start + int(local_left)
            right_index = int(right_index)
            affected_left.add(left_index)
            affected_right.add(right_index)
            pair_count += 1
            if left_labels[left_index] == right_labels[right_index]:
                agreement += 1
            else:
                conflicts += 1

    percentile_values = np.percentile(row_maxima, [50, 90, 95, 99])
    return NearDuplicateAudit(
        left_name=left_name,
        right_name=right_name,
        left_rows=len(left),
        right_rows=len(right),
        high_confidence_pairs=pair_count,
        affected_left_rows=tuple(sorted(affected_left)),
        affected_right_rows=tuple(sorted(affected_right)),
        label_agreement_pairs=agreement,
        label_conflict_pairs=conflicts,
        maximum_similarity=float(row_maxima.max(initial=0.0)),
        row_max_similarity_percentiles={
            name: float(value)
            for name, value in zip(("p50", "p90", "p95", "p99"), percentile_values)
        },
    )


def enforce_near_duplicate_safety(
    train_validation: NearDuplicateAudit,
    public_official: NearDuplicateAudit,
) -> None:
    """Apply the predeclared conflict and one-percent independence gates."""
    conflicts = train_validation.label_conflict_pairs + public_official.label_conflict_pairs
    if conflicts:
        raise NearDuplicateAuditError(
            f"Found {conflicts} high-confidence near-duplicate pairs with conflicting labels"
        )
    validation_share = len(train_validation.affected_right_rows) / max(
        train_validation.right_rows, 1
    )
    official_share = len(public_official.affected_right_rows) / max(
        public_official.right_rows, 1
    )
    if validation_share > 0.01:
        raise NearDuplicateAuditError(
            f"High-confidence matches affect {validation_share:.2%} of public validation rows"
        )
    if official_share > 0.01:
        raise NearDuplicateAuditError(
            f"High-confidence matches affect {official_share:.2%} of official evaluation rows"
        )


def remove_training_side_near_duplicates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    official: pd.DataFrame,
    train_validation: NearDuplicateAudit,
    public_official: NearDuplicateAudit,
    public_partition: pd.Series,
    public_source_index: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Remove only lower-precedence public rows identified by passing audits."""
    train_drop = set(train_validation.affected_left_rows)
    validation_drop: set[int] = set()
    for public_position in public_official.affected_left_rows:
        partition = str(public_partition.iloc[public_position])
        source_index = int(public_source_index.iloc[public_position])
        if partition == "public_train":
            train_drop.add(source_index)
        elif partition == "public_validation":
            validation_drop.add(source_index)
        else:
            raise ValueError(f"Unexpected public partition {partition!r}")
    train_clean = train.drop(index=sorted(train_drop)).reset_index(drop=True)
    validation_clean = validation.drop(index=sorted(validation_drop)).reset_index(drop=True)
    return train_clean, validation_clean, {
        "public_train_near": len(train_drop),
        "public_validation_near": len(validation_drop),
        "official_near": 0,
    }
