import numpy as np
import pandas as pd
import pytest

from olikbochon.metrics import (
    classification_metrics,
    predictions_from_label1,
    select_threshold,
    threshold_table,
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


def test_threshold_tie_breaking_is_deterministic() -> None:
    table = pd.DataFrame(
        [
            {"threshold": 0.40, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.60, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.50, "f1_label0": 0.8, "macro_f1": 0.6},
            {"threshold": 0.30, "f1_label0": 0.7, "macro_f1": 0.9},
        ]
    )
    assert select_threshold(table) == pytest.approx(0.40)


def test_threshold_macro_f1_is_first_tie_breaker() -> None:
    table = pd.DataFrame(
        [
            {"threshold": 0.49, "f1_label0": 0.8, "macro_f1": 0.7},
            {"threshold": 0.70, "f1_label0": 0.8, "macro_f1": 0.8},
        ]
    )
    assert select_threshold(table) == pytest.approx(0.70)
