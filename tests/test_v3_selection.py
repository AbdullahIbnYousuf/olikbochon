from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from olikbochon.v3_selection import (
    choose_arm,
    choose_deployment_threshold,
    write_safe_summary,
)


def test_arm_b_requires_all_three_promotion_conditions() -> None:
    truth = np.array([0, 1] * 50)
    weak = np.full(100, 0.5)
    strong = np.where(truth == 1, 0.9, 0.1)
    decision = choose_arm(truth, weak, strong)
    assert decision.selected_arm == "arm_b"
    assert decision.macro_gain_passed
    assert decision.collapse_passed
    assert decision.class0_guard_passed


def test_arm_a_is_locked_tie_breaker_and_collapse_guard() -> None:
    truth = np.array([0, 1] * 50)
    equal = np.where(truth == 1, 0.9, 0.1)
    assert choose_arm(truth, equal, equal).selected_arm == "arm_a"
    collapsed = np.zeros(100)
    assert choose_arm(truth, np.full(100, 0.5), collapsed).selected_arm == "arm_a"


def test_threshold_deployment_guards_range_gain_and_collapse() -> None:
    truth = np.array([0, 1] * 50)
    probabilities = np.where(truth == 1, 0.55, 0.45)
    decision = choose_deployment_threshold(truth, probabilities)
    assert 0.40 <= decision.tuned_threshold <= 0.60
    assert decision.deployed_threshold in {0.5, decision.tuned_threshold}


def test_safe_summary_rejects_row_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    with pytest.raises(ValueError, match="row-level"):
        write_safe_summary(path, {"predictions": [0, 1]})
    write_safe_summary(path, {"prediction_distribution": {0: 1, 1: 1}})
    assert json.loads(path.read_text(encoding="utf-8"))["prediction_distribution"] == {
        "0": 1,
        "1": 1,
    }
