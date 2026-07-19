"""Aggregate-only calibration diagnostics for authenticated V4 fold checkpoints."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import classification_metrics, predictions_from_label1
from .v4_validation import select_threshold


DIAGNOSTIC_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _threshold_record(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = predictions_from_label1(probabilities, threshold)
    counts = np.bincount(predictions, minlength=2)
    return {
        "threshold": float(threshold),
        **classification_metrics(truth, predictions),
        "predicted_label1_prevalence": float(predictions.mean()),
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
    }


def probability_diagnostics(
    truth: Sequence[int], probabilities_label1: Sequence[float]
) -> dict[str, Any]:
    """Summarize calibration and the frozen grid without retaining row-level values."""
    labels = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities_label1, dtype=np.float64)
    if labels.size == 0 or labels.shape != probabilities.shape:
        raise ValueError("Truth and probabilities must be nonempty and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Diagnostic truth must be binary")
    if not np.isfinite(probabilities).all() or not np.logical_and(
        probabilities >= 0.0, probabilities <= 1.0
    ).all():
        raise ValueError("Diagnostic probabilities must be finite and between zero and one")
    threshold_050 = _threshold_record(labels, probabilities, 0.50)
    threshold_054 = _threshold_record(labels, probabilities, 0.54)
    best_threshold, best_metrics = select_threshold(labels, probabilities)
    best_predictions = predictions_from_label1(probabilities, best_threshold)
    best_counts = np.bincount(best_predictions, minlength=2)
    best_record = {
        "threshold": best_threshold,
        **best_metrics,
        "predicted_label1_prevalence": float(best_predictions.mean()),
        "prediction_counts": {"0": int(best_counts[0]), "1": int(best_counts[1])},
    }
    macro_gap = float(best_metrics["macro_f1"] - threshold_054["macro_f1"])
    quantiles = np.quantile(probabilities, DIAGNOSTIC_QUANTILES)
    return {
        "row_count": int(labels.size),
        "probability_minimum": float(probabilities.min()),
        "probability_maximum": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "probability_median": float(np.median(probabilities)),
        "probability_std": float(probabilities.std(ddof=0)),
        "probability_quantiles": {
            f"{quantile:.2f}": float(value)
            for quantile, value in zip(DIAGNOSTIC_QUANTILES, quantiles)
        },
        "true_label1_prevalence": float(labels.mean()),
        "brier_score": float(np.mean(np.square(probabilities - labels))),
        "threshold_050": threshold_050,
        "threshold_054": threshold_054,
        "best_frozen_grid": best_record,
        "threshold_054_macro_f1_gap_from_best": macro_gap,
        "threshold_054_near_optimal": (
            abs(best_threshold - 0.54) <= 0.02 and macro_gap <= 0.02
        ),
    }


def selected_epoch_distribution(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count selected epochs without retaining any row-level training data."""
    counts = Counter(int(record["selected_epoch"]) for record in records)
    return {str(epoch): counts[epoch] for epoch in sorted(counts)}
