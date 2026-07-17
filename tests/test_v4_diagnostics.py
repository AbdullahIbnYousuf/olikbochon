from __future__ import annotations

import numpy as np
import pytest

from olikbochon.v4_diagnostics import probability_diagnostics, selected_epoch_distribution


def test_probability_diagnostics_reports_frozen_grid_and_calibration() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.10, 0.40, 0.60, 0.90], dtype=np.float64)
    result = probability_diagnostics(truth, probabilities)
    assert result["row_count"] == 4
    assert result["probability_median"] == 0.5
    assert result["true_label1_prevalence"] == 0.5
    assert result["brier_score"] == pytest.approx(0.085)
    assert result["threshold_050"]["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["threshold_054"]["prediction_counts"] == {"0": 2, "1": 2}
    assert 0.20 <= result["best_frozen_grid"]["threshold"] <= 0.80
    assert set(result["probability_quantiles"]) == {
        "0.01",
        "0.05",
        "0.10",
        "0.25",
        "0.50",
        "0.75",
        "0.90",
        "0.95",
        "0.99",
    }


def test_selected_epoch_distribution_is_numeric_and_sorted() -> None:
    records = [{"selected_epoch": 3}, {"selected_epoch": 1}, {"selected_epoch": 3}]
    assert selected_epoch_distribution(records) == {"1": 1, "3": 2}
