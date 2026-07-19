"""Shared, leakage-resistant protocol for V7 compatible-fold experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data_loading import DataValidationError, validate_labeled_frame
from .metrics import classification_metrics, subgroup_metrics
from .v5_lexical import build_v5_folds
from .v5_normalization import normalize_v5


OUTER_SEEDS = (17, 29, 43)
OUTER_FOLDS = 5
INNER_FOLDS = 3
PROBABILITY_COLUMNS = ("probability_label_0", "probability_label_1")
ABSENT_CONTEXT_SENTINELS = frozenset(
    {"", "[null]", "null", "none", "n/a", "nan", "<na>"}
)


@dataclass(frozen=True)
class InnerFold:
    inner_fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class FitScope:
    """Stable-ID provenance for one fit or selection operation."""

    role: str
    training_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    test_row_count: int = 0

    def validate(self) -> None:
        train = set(self.training_row_ids)
        validation = set(self.validation_row_ids)
        if not train or not validation:
            raise DataValidationError(f"{self.role} requires nonempty train and validation IDs")
        if len(train) != len(self.training_row_ids):
            raise DataValidationError(f"{self.role} has duplicate training row IDs")
        if len(validation) != len(self.validation_row_ids):
            raise DataValidationError(f"{self.role} has duplicate validation row IDs")
        if train & validation:
            raise DataValidationError(f"{self.role} leaks validation IDs into training")
        if self.test_row_count != 0:
            raise DataValidationError(f"{self.role} must not access competition test rows")


@dataclass(frozen=True)
class SelectionAudit:
    """Prove a hyperparameter/threshold decision used inner rows only."""

    selection_kind: str
    inner_training_row_ids: tuple[str, ...]
    inner_validation_row_ids: tuple[str, ...]
    outer_validation_row_ids: tuple[str, ...] = ()
    test_row_count: int = 0

    def validate(self) -> None:
        allowed = {
            "retrieval_acceptance_threshold",
            "classification_threshold",
            "model_hyperparameter",
            "fusion_regularization",
        }
        if self.selection_kind not in allowed:
            raise DataValidationError(f"Unknown nested selection kind: {self.selection_kind}")
        FitScope(
            role=f"{self.selection_kind}_inner_selection",
            training_row_ids=self.inner_training_row_ids,
            validation_row_ids=self.inner_validation_row_ids,
            test_row_count=self.test_row_count,
        ).validate()
        if self.outer_validation_row_ids:
            raise DataValidationError(
                f"{self.selection_kind} must not use untouched outer-validation rows"
            )


def has_context(value: Any) -> bool:
    """Return schema-aware context presence without truthy-string mistakes."""
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False
    if not isinstance(value, str):
        value = str(value)
    return value.strip().casefold() not in ABSENT_CONTEXT_SENTINELS


def stable_training_row_ids(frame: pd.DataFrame) -> tuple[str, ...]:
    """Build content-derived IDs; fail instead of silently using DataFrame position."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    identifiers: list[str] = []
    for row in validated.itertuples(index=False):
        payload = {
            "context_present": has_context(row.context),
            "context": normalize_v5(row.context) if has_context(row.context) else "",
            "prompt_bn": normalize_v5(row.prompt_bn),
            "response_bn": normalize_v5(row.response_bn),
            "label": int(row.label),
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        identifiers.append("train-" + hashlib.sha256(encoded).hexdigest())
    if len(set(identifiers)) != len(identifiers):
        raise DataValidationError(
            "Content-derived training row IDs are not unique; provide an explicit stable ID"
        )
    return tuple(identifiers)


def build_common_fold_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the one frozen strengthened 3x5 outer-fold table for all V7 systems."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    row_ids = np.asarray(stable_training_row_ids(validated), dtype=object)
    presence = np.asarray([has_context(value) for value in validated["context"]], dtype=bool)
    repeated = build_v5_folds(validated)
    if tuple(repeated.seeds) != OUTER_SEEDS or len(repeated.folds) != 15:
        raise DataValidationError("Strengthened fold implementation violates the V7 freeze")
    groups = np.asarray(repeated.audit.group_ids, dtype=object)
    labels = validated["label"].to_numpy(dtype=np.int64)
    records: list[pd.DataFrame] = []
    for split in repeated.folds:
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        records.append(
            pd.DataFrame(
                {
                    "row_id": row_ids[validation],
                    "label": labels[validation],
                    "context_present": presence[validation],
                    "group_id": groups[validation],
                    "seed": int(split.seed),
                    "outer_fold": int(split.fold),
                }
            )
        )
    result = pd.concat(records, ignore_index=True).sort_values(
        ["seed", "outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_common_fold_assignments(result, row_ids)
    return result


def validate_common_fold_assignments(assignments: pd.DataFrame, row_ids: Sequence[str]) -> None:
    expected = ("row_id", "label", "context_present", "group_id", "seed", "outer_fold")
    if tuple(assignments.columns) != expected:
        raise DataValidationError(f"Common fold columns must be {expected}")
    expected_ids = set(map(str, row_ids))
    if set(assignments["seed"].astype(int)) != set(OUTER_SEEDS):
        raise DataValidationError("Common folds must contain exactly seeds 17, 29, and 43")
    for seed, seed_rows in assignments.groupby("seed", sort=True):
        if seed_rows["row_id"].duplicated().any():
            raise DataValidationError(f"Seed {seed} has duplicate row IDs")
        if set(seed_rows["row_id"].astype(str)) != expected_ids:
            raise DataValidationError(f"Seed {seed} does not cover every official row once")
        if set(seed_rows["outer_fold"].astype(int)) != set(range(1, OUTER_FOLDS + 1)):
            raise DataValidationError(f"Seed {seed} does not contain five outer folds")
        for fold, fold_rows in seed_rows.groupby("outer_fold", sort=True):
            if set(fold_rows["label"].astype(int)) != {0, 1}:
                raise DataValidationError(f"Seed {seed}, fold {fold} lacks a label")


def outer_indices(
    assignments: pd.DataFrame, row_ids: Sequence[str], *, seed: int, outer_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a frozen split by stable ID, independent of DataFrame row position."""
    mapping = {str(row_id): index for index, row_id in enumerate(row_ids)}
    if len(mapping) != len(row_ids):
        raise DataValidationError("Dataset row IDs must be unique")
    selected = assignments.loc[
        (assignments["seed"] == seed) & (assignments["outer_fold"] == outer_fold),
        "row_id",
    ].astype(str)
    if selected.empty or any(row_id not in mapping for row_id in selected):
        raise DataValidationError("Fold assignment contains missing or unknown stable IDs")
    validation = np.asarray([mapping[row_id] for row_id in selected], dtype=np.int64)
    train = np.asarray(sorted(set(range(len(row_ids))) - set(validation)), dtype=np.int64)
    return train, validation


def make_inner_grouped_folds(
    labels: Sequence[int],
    group_ids: Sequence[str],
    outer_train_indices: Sequence[int],
    *,
    seed: int,
    outer_fold: int,
) -> tuple[InnerFold, ...]:
    """Generate nested grouped folds using only the current outer-training partition."""
    truth = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(group_ids, dtype=object)
    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    if len(set(outer_train.tolist())) != len(outer_train):
        raise DataValidationError("Outer-training indices must be unique")
    local_truth = truth[outer_train]
    local_groups = groups[outer_train]
    splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=int(seed) * 100 + int(outer_fold),
    )
    coverage = np.zeros(len(outer_train), dtype=np.int64)
    result: list[InnerFold] = []
    for number, (local_train, local_validation) in enumerate(
        splitter.split(np.arange(len(outer_train)), local_truth, local_groups), start=1
    ):
        if set(local_groups[local_train]) & set(local_groups[local_validation]):
            raise DataValidationError("Inner grouped split leaks a group")
        if set(np.unique(local_truth[local_validation])) != {0, 1}:
            raise DataValidationError("Inner validation fold lacks a label")
        coverage[local_validation] += 1
        result.append(
            InnerFold(
                number,
                tuple(int(value) for value in outer_train[local_train]),
                tuple(int(value) for value in outer_train[local_validation]),
            )
        )
    if not np.all(coverage == 1):
        raise DataValidationError("Inner folds do not cover outer-training rows exactly once")
    return tuple(result)


def validate_probability_frame(
    frame: pd.DataFrame,
    expected_row_ids: Iterable[str],
    *,
    require_seed: bool = True,
) -> None:
    """Validate explicit binary orientation, stable-ID coverage, and numeric bounds."""
    required = {"row_id", "probability_label_0", "probability_label_1"}
    if require_seed:
        required.add("seed")
    if not required.issubset(frame.columns):
        raise DataValidationError(f"Probability artifact is missing {sorted(required-frame.columns)}")
    if any(name in frame.columns for name in ("probability", "probability_label1")):
        raise DataValidationError("V7 probability columns must use explicit label orientation")
    values = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise DataValidationError("Probabilities must be finite values in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
        raise DataValidationError("Binary probability columns must sum to one")
    expected = set(map(str, expected_row_ids))
    if require_seed:
        for seed in OUTER_SEEDS:
            selected = frame.loc[frame["seed"] == seed, "row_id"].astype(str)
            if selected.duplicated().any() or set(selected) != expected:
                raise DataValidationError(f"Seed {seed} lacks exact one-time OOF coverage")
    elif frame["row_id"].astype(str).duplicated().any() or set(frame["row_id"].astype(str)) != expected:
        raise DataValidationError("Probability artifact lacks exact stable-ID coverage")


def metric_record(
    labels: Sequence[int],
    probabilities_label_1: Sequence[float],
    predictions: Sequence[int],
    context_present: Sequence[bool],
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities_label_1, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if not (truth.shape == probabilities.shape == predicted.shape):
        raise DataValidationError("Metric arrays must be aligned")
    counts = np.bincount(predicted, minlength=2)
    return {
        **classification_metrics(truth, predicted),
        "route_metrics": subgroup_metrics(truth, predicted, context_present),
        "brier_score": float(np.mean(np.square(probabilities - truth))),
        "probability_mean": float(probabilities.mean()),
        "probability_std": float(probabilities.std(ddof=0)),
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "maximum_predicted_class_share": float(counts.max() / counts.sum()),
    }


def calibration_table(
    labels: Sequence[int], probabilities_label_1: Sequence[float], bins: int = 10
) -> list[dict[str, Any]]:
    truth = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities_label_1, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    records: list[dict[str, Any]] = []
    for index in range(bins):
        mask = indices == index
        records.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_probability_label_1": (
                    float(probabilities[mask].mean()) if mask.any() else None
                ),
                "observed_label_1_rate": float(truth[mask].mean()) if mask.any() else None,
            }
        )
    return records
