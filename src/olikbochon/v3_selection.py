"""Deterministic Version 3 metrics, arm selection, and threshold deployment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    classification_metrics,
    fold_mean_std,
    prediction_collapse_warning,
    predictions_from_label1,
    select_threshold,
    subgroup_metrics,
    threshold_table,
)


@dataclass(frozen=True)
class ArmDecision:
    selected_arm: str
    arm_b_macro_gain: float
    macro_gain_passed: bool
    collapse_passed: bool
    class0_guard_passed: bool
    reason: str


@dataclass(frozen=True)
class ThresholdDecision:
    tuned_threshold: float
    deployed_threshold: float
    macro_gain: float
    gain_passed: bool
    range_passed: bool
    collapse_passed: bool
    reason: str


def prediction_distribution(predictions: np.ndarray) -> dict[int, int]:
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=2)
    return {0: int(counts[0]), 1: int(counts[1])}


def evaluate_oof(
    truth: np.ndarray,
    probabilities: np.ndarray,
    fold_metrics: list[dict[str, Any]],
    has_context: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return aggregate-only official OOF reporting at one threshold."""
    predictions = predictions_from_label1(probabilities, threshold)
    warning = prediction_collapse_warning(predictions, name=f"OOF threshold {threshold:.2f}")
    return {
        "threshold": float(threshold),
        "metrics": classification_metrics(truth, predictions),
        "fold_summary": fold_mean_std(fold_metrics),
        "prediction_distribution": prediction_distribution(predictions),
        "context_metrics": subgroup_metrics(truth, predictions, has_context),
        "collapse_warning": warning,
    }


def choose_arm(
    truth: np.ndarray,
    arm_a_probabilities: np.ndarray,
    arm_b_probabilities: np.ndarray,
) -> ArmDecision:
    """Apply the locked three-condition Arm B promotion rule."""
    a_predictions = predictions_from_label1(arm_a_probabilities, 0.5)
    b_predictions = predictions_from_label1(arm_b_probabilities, 0.5)
    a = classification_metrics(truth, a_predictions)
    b = classification_metrics(truth, b_predictions)
    gain = float(b["macro_f1"] - a["macro_f1"])
    gain_passed = gain >= 0.01 - 1e-12
    collapse_passed = prediction_collapse_warning(b_predictions, name="Arm B OOF") is None
    class0_passed = float(b["f1_label0"]) >= float(a["f1_label0"]) - 0.02 - 1e-12
    selected = "arm_b" if gain_passed and collapse_passed and class0_passed else "arm_a"
    reason = (
        "Arm B satisfied all locked promotion conditions"
        if selected == "arm_b"
        else "Arm A retained because one or more Arm B promotion conditions failed"
    )
    return ArmDecision(selected, gain, gain_passed, collapse_passed, class0_passed, reason)


def choose_deployment_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
) -> ThresholdDecision:
    """Tune on selected-arm OOF and deploy only under the three safety guards."""
    table = threshold_table(truth, probabilities)
    tuned = select_threshold(table, strategy="macro_f1_oof")
    fixed_predictions = predictions_from_label1(probabilities, 0.5)
    tuned_predictions = predictions_from_label1(probabilities, tuned)
    fixed = classification_metrics(truth, fixed_predictions)
    selected = classification_metrics(truth, tuned_predictions)
    gain = float(selected["macro_f1"] - fixed["macro_f1"])
    gain_passed = gain >= 0.01 - 1e-12
    range_passed = 0.40 <= tuned <= 0.60
    collapse_passed = (
        prediction_collapse_warning(tuned_predictions, name="selected threshold OOF") is None
    )
    deployed = tuned if gain_passed and range_passed and collapse_passed else 0.5
    reason = (
        "Tuned threshold satisfied all deployment guards"
        if deployed == tuned
        else "Fixed 0.50 retained because one or more deployment guards failed"
    )
    return ThresholdDecision(
        tuned, deployed, gain, gain_passed, range_passed, collapse_passed, reason
    )


def _safe_value(value: Any) -> Any:
    if is_dataclass(value):
        return _safe_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsafe run-summary value type: {type(value).__name__}")


def write_safe_summary(path: Path, summary: dict[str, Any]) -> None:
    """Serialize aggregate/configuration values only; reject unknown object types."""
    forbidden_keys = {"id", "ids", "text", "texts", "probabilities", "predictions", "rows"}

    def audit_keys(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in forbidden_keys:
                    raise ValueError(f"Unsafe row-level run-summary key: {key!r}")
                audit_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                audit_keys(item)

    audit_keys(summary)
    safe = _safe_value(summary)
    Path(path).write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
