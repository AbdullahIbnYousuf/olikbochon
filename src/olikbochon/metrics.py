"""Metrics, threshold selection, and honest nested threshold validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from olikbochon.modeling import RANDOM_STATE, build_model, label_probability


THRESHOLDS = np.arange(20, 81, dtype=np.int64) / 100.0


@dataclass(frozen=True)
class OOFResult:
    """Aggregate outputs retained in memory for safe metric reporting."""

    probabilities_label1: np.ndarray
    predictions: np.ndarray
    fold_metrics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class NestedCVResult:
    """Untouched outer-fold predictions from nested threshold selection."""

    predictions: np.ndarray
    selected_thresholds: tuple[float, ...]
    fold_metrics: tuple[dict[str, Any], ...]


def predictions_from_label1(probabilities: Sequence[float], threshold: float) -> np.ndarray:
    """Apply the competition label rule to P(label=1)."""
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    return (probabilities_array >= float(threshold)).astype(np.int64)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Calculate all required aggregate metrics with explicit label direction."""
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    return {
        "f1_label0": float(f1_score(truth, predicted, pos_label=0, zero_division=0)),
        "f1_label1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).astype(int).tolist(),
    }


def threshold_table(y_true: Sequence[int], probabilities_label1: Sequence[float]) -> pd.DataFrame:
    """Evaluate the locked threshold grid without retaining row-level outputs."""
    rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        metrics = classification_metrics(
            y_true, predictions_from_label1(probabilities_label1, float(threshold))
        )
        rows.append(
            {
                "threshold": float(threshold),
                "f1_label0": metrics["f1_label0"],
                "f1_label1": metrics["f1_label1"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame) -> float:
    """Select a threshold using the locked deterministic four-level ordering."""
    required = {"threshold", "f1_label0", "macro_f1"}
    if not required.issubset(table.columns) or table.empty:
        raise ValueError(f"Threshold table must be nonempty and contain {sorted(required)}")
    ranked = table.assign(distance=(table["threshold"] - 0.5).abs()).sort_values(
        ["f1_label0", "macro_f1", "distance", "threshold"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["threshold"])


def standard_oof_predictions(
    texts: Sequence[str], y_true: Sequence[int], threshold: float = 0.5
) -> OOFResult:
    """Generate standard 5-fold OOF probabilities using the locked outer split."""
    texts_array = np.asarray(texts, dtype=object)
    truth = np.asarray(y_true, dtype=np.int64)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probabilities = np.empty(len(truth), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, (train_index, valid_index) in enumerate(cv.split(texts_array, truth), start=1):
        model = build_model()
        model.fit(texts_array[train_index], truth[train_index])
        fold_probabilities = label_probability(model, texts_array[valid_index], label=1)
        probabilities[valid_index] = fold_probabilities
        fold_predictions = predictions_from_label1(fold_probabilities, threshold)
        fold_rows.append(
            {
                "fold": fold_index,
                "threshold": float(threshold),
                **classification_metrics(truth[valid_index], fold_predictions),
            }
        )

    predictions = predictions_from_label1(probabilities, threshold)
    return OOFResult(probabilities, predictions, tuple(fold_rows))


def nested_threshold_predictions(texts: Sequence[str], y_true: Sequence[int]) -> NestedCVResult:
    """Evaluate threshold selection with inner OOF tuning and untouched outer folds."""
    texts_array = np.asarray(texts, dtype=object)
    truth = np.asarray(y_true, dtype=np.int64)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    outer_predictions = np.empty(len(truth), dtype=np.int64)
    thresholds: list[float] = []
    fold_rows: list[dict[str, Any]] = []

    for outer_zero_index, (outer_train, outer_valid) in enumerate(
        outer_cv.split(texts_array, truth)
    ):
        inner_truth = truth[outer_train]
        inner_texts = texts_array[outer_train]
        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE + outer_zero_index,
        )
        inner_probabilities = np.empty(len(inner_truth), dtype=np.float64)

        for inner_train, inner_valid in inner_cv.split(inner_texts, inner_truth):
            inner_model = build_model()
            inner_model.fit(inner_texts[inner_train], inner_truth[inner_train])
            inner_probabilities[inner_valid] = label_probability(
                inner_model, inner_texts[inner_valid], label=1
            )

        selected = select_threshold(threshold_table(inner_truth, inner_probabilities))
        thresholds.append(selected)
        outer_model = build_model()
        outer_model.fit(texts_array[outer_train], truth[outer_train])
        valid_probabilities = label_probability(outer_model, texts_array[outer_valid], label=1)
        valid_predictions = predictions_from_label1(valid_probabilities, selected)
        outer_predictions[outer_valid] = valid_predictions
        fold_rows.append(
            {
                "fold": outer_zero_index + 1,
                "threshold": selected,
                **classification_metrics(truth[outer_valid], valid_predictions),
            }
        )

    return NestedCVResult(outer_predictions, tuple(thresholds), tuple(fold_rows))


def fold_mean_std(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize scalar fold metrics without row-level outputs."""
    names = ("f1_label0", "f1_label1", "macro_f1", "accuracy")
    return {
        name: {
            "mean": float(np.mean([float(row[name]) for row in fold_metrics])),
            "std": float(np.std([float(row[name]) for row in fold_metrics], ddof=0)),
        }
        for name in names
    }


def subgroup_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], has_context: Sequence[bool]
) -> dict[str, dict[str, Any]]:
    """Report safe aggregate metrics by normalized context regime."""
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    presence = np.asarray(has_context, dtype=bool)
    result: dict[str, dict[str, Any]] = {}
    for name, mask in (("context_present", presence), ("context_absent", ~presence)):
        labels = np.unique(truth[mask])
        row: dict[str, Any] = {"count": int(mask.sum()), "both_classes": len(labels) == 2}
        if len(labels) == 2:
            metrics = classification_metrics(truth[mask], predicted[mask])
            row.update(
                {
                    "f1_label0": metrics["f1_label0"],
                    "macro_f1": metrics["macro_f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                }
            )
        result[name] = row
    return result
