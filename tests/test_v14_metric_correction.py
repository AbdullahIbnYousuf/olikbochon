from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from olikbochon.champion_submission import validate_champion_test_frame
from olikbochon.submission import build_submission
from olikbochon.v14_evaluation import _apply_candidate
from olikbochon.v14_metrics import (
    hallucinated_f1,
    predictions_from_label1,
    select_route_thresholds,
    select_threshold,
)


def test_hallucinated_f1_uses_label_zero_as_positive() -> None:
    truth = np.asarray([0, 0, 0, 1, 1])
    prediction = np.asarray([0, 0, 1, 0, 1])
    assert hallucinated_f1(truth, prediction) == f1_score(
        truth, prediction, pos_label=0, zero_division=0
    )
    assert hallucinated_f1(truth, prediction) != f1_score(
        truth, prediction, pos_label=1, zero_division=0
    )


def test_probability_direction_is_label_one_at_or_above_threshold() -> None:
    assert predictions_from_label1(np.asarray([0.49, 0.50, 0.90]), 0.50).tolist() == [0, 1, 1]


def test_threshold_selection_uses_only_supplied_inner_rows() -> None:
    inner_truth = np.asarray([0, 0, 1, 1])
    inner_probability = np.asarray([0.20, 0.30, 0.40, 0.90])
    selected = select_threshold(inner_truth, inner_probability)["selected"]["threshold"]
    outer_truth = np.asarray([0, 0])
    outer_probability = np.asarray([0.80, 0.85])
    assert select_threshold(inner_truth, inner_probability)["selected"]["threshold"] == selected
    assert select_threshold(
        np.concatenate((inner_truth, outer_truth)),
        np.concatenate((inner_probability, outer_probability)),
    )["selected"]["threshold"] != selected


def test_route_threshold_selection_and_application() -> None:
    truth = np.asarray([0, 1, 0, 1])
    presence = np.asarray([True, True, False, False])
    present = np.asarray([0.10, 0.80, np.nan, np.nan])
    absent = np.asarray([np.nan, np.nan, 0.20, 0.90])
    selected = select_route_thresholds(truth, presence, present, absent)["selected"]
    probabilities = {"v4a": present, "i": absent, "r": absent, "u": absent}
    prediction = _apply_candidate(
        "Y1", presence, probabilities, np.zeros(4, dtype=np.int64), selected
    )
    assert prediction.tolist() == truth.tolist()


def test_submission_is_deterministic_and_contains_no_test_text() -> None:
    ids = pd.Series(["a", "b", "c"])
    labels = np.asarray([0, 1, 0])
    first = build_submission(ids, labels)
    second = build_submission(ids, labels)
    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == ["id", "label"]
    assert "context" not in first and "prompt_bn" not in first and "response_bn" not in first


def test_threshold_rejects_nonfinite_probability() -> None:
    with pytest.raises(ValueError, match="finite"):
        predictions_from_label1(np.asarray([0.2, np.nan]), 0.5)


def test_competition_test_contract_has_no_label_input() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"id-{index}" for index in range(2_516)],
            "context": [None] * 2_516,
            "prompt_bn": ["synthetic"] * 2_516,
            "response_bn": ["synthetic"] * 2_516,
        }
    )
    validated = validate_champion_test_frame(frame)
    assert "label" not in validated.columns
    with pytest.raises(ValueError, match="Expected test columns"):
        validate_champion_test_frame(frame.assign(label=0))
