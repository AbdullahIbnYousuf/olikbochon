import numpy as np
import pandas as pd
import pytest

from olikbochon.metrics import (
    CLASS0_F1_OOF_EXPERIMENTAL,
    DEFAULT_THRESHOLD_STRATEGY,
    FIXED_050,
    MACRO_F1_OOF,
    classification_metrics,
    prediction_collapse_warning,
    predictions_from_label1,
    select_threshold,
    threshold_table,
    trivial_predictor_metrics,
)


def test_class_specific_and_macro_f1() -> None:
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["f1_label0"] == pytest.approx(2 / 3)
    assert metrics["f1_label1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_probability_rule_and_threshold_table() -> None:
    probabilities = np.array([0.19, 0.50, 0.51, 0.81])
    assert predictions_from_label1(probabilities, 0.5).tolist() == [0, 1, 1, 1]
    table = threshold_table([0, 0, 1, 1], probabilities)
    assert len(table) == 61
    assert table["threshold"].iloc[0] == pytest.approx(0.2)
    assert table["threshold"].iloc[-1] == pytest.approx(0.8)


def test_class0_threshold_tie_breaking_is_deterministic() -> None:
    table = pd.DataFrame(
        [
            {"threshold": 0.40, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.60, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.50, "f1_label0": 0.8, "macro_f1": 0.6},
            {"threshold": 0.30, "f1_label0": 0.7, "macro_f1": 0.9},
        ]
    )
    assert select_threshold(table, CLASS0_F1_OOF_EXPERIMENTAL) == pytest.approx(0.40)


def test_class0_threshold_uses_macro_f1_as_first_tie_breaker() -> None:
    table = pd.DataFrame(
        [
            {"threshold": 0.49, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.70, "f1_label0": 0.8, "macro_f1": 0.8},
        ]
    )
    assert select_threshold(table, CLASS0_F1_OOF_EXPERIMENTAL) == pytest.approx(0.70)


def test_macro_f1_threshold_selection() -> None:
    table = pd.DataFrame(
        [
            {"threshold": 0.40, "f1_label0": 0.9, "macro_f1": 0.70},
            {"threshold": 0.55, "f1_label0": 0.6, "macro_f1": 0.75},
        ]
    )
    assert select_threshold(table, MACRO_F1_OOF) == pytest.approx(0.55)


def test_macro_f1_tie_breaking_is_deterministic() -> None:
    table = pd.DataFrame(
        [
            {"threshold": 0.45, "f1_label0": 0.70, "macro_f1": 0.80},
            {"threshold": 0.60, "f1_label0": 0.75, "macro_f1": 0.80},
            {"threshold": 0.40, "f1_label0": 0.75, "macro_f1": 0.80},
        ]
    )
    assert select_threshold(table, MACRO_F1_OOF) == pytest.approx(0.40)


def test_default_threshold_strategy_is_macro_f1_oof() -> None:
    assert DEFAULT_THRESHOLD_STRATEGY == MACRO_F1_OOF


def test_fixed_strategy_returns_clean_reference_threshold() -> None:
    table = pd.DataFrame(
        [{"threshold": 0.8, "f1_label0": 1.0, "macro_f1": 1.0}]
    )
    assert select_threshold(table, FIXED_050) == pytest.approx(0.5)


def test_all_zero_baseline_calculation() -> None:
    baselines = trivial_predictor_metrics([0, 0, 1, 1, 1])
    all_zero = baselines["predict_all_0"]
    assert all_zero["predicted_label"] == 0
    assert all_zero["f1_label0"] == pytest.approx(4 / 7)
    assert all_zero["f1_label1"] == 0
    assert all_zero["macro_f1"] == pytest.approx(2 / 7)
    assert all_zero["accuracy"] == pytest.approx(0.4)
    assert all_zero["confusion_matrix"] == [[2, 0], [3, 0]]
    assert baselines["majority_class"] == baselines["predict_all_1"]


def test_prediction_collapse_warning() -> None:
    warning = prediction_collapse_warning([0] * 91 + [1] * 9, name="experimental")
    assert warning is not None
    assert "prediction collapse" in warning
    assert "class-specific F1" in warning
    assert prediction_collapse_warning([0] * 90 + [1] * 10, name="balanced") is None
