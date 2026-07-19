"""Competition-metric primitives for the V14 correction cycle."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


THRESHOLD_GRID = tuple(float(value) for value in np.arange(0.10, 0.901, 0.05))


def hallucinated_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    """Return the official binary F1 with hallucination label 0 as positive."""
    return float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))


def predictions_from_label1(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Convert P(label 1) to labels; values below threshold are hallucination label 0."""
    values = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(values).all() or not 0.0 <= threshold <= 1.0:
        raise ValueError("V14 probabilities and threshold must be finite and valid")
    return (values >= threshold).astype(np.int64)


def metric_record(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != prediction.shape or not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("V14 metrics require aligned binary arrays")
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    counts = np.bincount(prediction, minlength=2)
    return {
        "label0_precision": float(precision_score(truth, prediction, pos_label=0, zero_division=0)),
        "label0_recall": float(recall_score(truth, prediction, pos_label=0, zero_division=0)),
        "label0_f1": hallucinated_f1(truth, prediction),
        "label1_f1": float(f1_score(truth, prediction, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "predicted_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "predicted_label0_share": float(counts[0] / len(prediction)),
    }


def select_threshold(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    records = []
    for threshold in THRESHOLD_GRID:
        metrics = metric_record(truth, predictions_from_label1(probability, threshold))
        records.append({"threshold": threshold, **metrics})
    selected = max(
        records,
        key=lambda record: (
            record["label0_f1"],
            record["label0_precision"],
            -abs(record["threshold"] - 0.50),
            -record["threshold"],
        ),
    )
    return {"selected": selected, "grid": records}


def select_route_thresholds(
    truth: np.ndarray,
    presence: np.ndarray,
    present_probability: np.ndarray,
    absent_probability: np.ndarray,
) -> dict[str, Any]:
    records = []
    for present_threshold in THRESHOLD_GRID:
        for absent_threshold in THRESHOLD_GRID:
            prediction = np.empty(len(truth), dtype=np.int64)
            prediction[presence] = predictions_from_label1(
                present_probability[presence], present_threshold
            )
            prediction[~presence] = predictions_from_label1(
                absent_probability[~presence], absent_threshold
            )
            records.append(
                {
                    "present_threshold": present_threshold,
                    "absent_threshold": absent_threshold,
                    **metric_record(truth, prediction),
                }
            )
    selected = max(
        records,
        key=lambda record: (
            record["label0_f1"],
            record["label0_precision"],
            -(
                abs(record["present_threshold"] - 0.50)
                + abs(record["absent_threshold"] - 0.50)
            ),
            -record["present_threshold"],
            -record["absent_threshold"],
        ),
    )
    return {"selected": selected, "grid_size": len(records)}


def select_absent_threshold_with_hard_present(
    truth: np.ndarray,
    presence: np.ndarray,
    hard_present: np.ndarray,
    absent_probability: np.ndarray,
) -> dict[str, Any]:
    records = []
    for threshold in THRESHOLD_GRID:
        prediction = np.empty(len(truth), dtype=np.int64)
        prediction[presence] = hard_present[presence]
        prediction[~presence] = predictions_from_label1(absent_probability[~presence], threshold)
        records.append({"absent_threshold": threshold, **metric_record(truth, prediction)})
    selected = max(
        records,
        key=lambda record: (
            record["label0_f1"],
            record["label0_precision"],
            -abs(record["absent_threshold"] - 0.50),
            -record["absent_threshold"],
        ),
    )
    return {"selected": selected, "grid": records}
