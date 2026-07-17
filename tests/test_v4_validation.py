from __future__ import annotations

import numpy as np
import pytest

from olikbochon.v3_data import GroupAudit
from olikbochon.v4_validation import (
    CANDIDATE_ORDER,
    THRESHOLD_GRID,
    VALIDATION_SEEDS,
    CandidateEvaluation,
    choose_candidate,
    fold_metric_record,
    make_repeated_grouped_folds,
    route_threshold_diagnostics,
    select_threshold,
    summarize_fold_metrics,
)


def audit_for(groups: list[str]) -> GroupAudit:
    counts = {group: groups.count(group) for group in set(groups)}
    return GroupAudit(
        tuple(groups),
        len(counts),
        sum(count > 1 for count in counts.values()),
        max(counts.values()),
        0,
        0,
        0,
        0,
    )


def test_repeated_grouped_folds_are_deterministic_and_isolated() -> None:
    labels = np.asarray([0, 1] * 30)
    groups = [f"group-{index // 2:02d}" for index in range(60)]
    first = make_repeated_grouped_folds(labels, audit_for(groups))
    second = make_repeated_grouped_folds(labels, audit_for(groups))
    assert first == second
    assert first.seeds == VALIDATION_SEEDS
    assert len(first.folds) == 15
    group_array = np.asarray(groups)
    for fold in first.folds:
        assert not (
            set(group_array[list(fold.train_indices)])
            & set(group_array[list(fold.validation_indices)])
        )


def test_invalid_grouped_fold_does_not_fall_back_to_random_split() -> None:
    with pytest.raises(Exception, match="groups are available"):
        make_repeated_grouped_folds([0, 1, 0, 1], audit_for(["a", "b", "c", "d"]))


def test_threshold_grid_is_exact_and_selection_rejects_collapse() -> None:
    assert THRESHOLD_GRID[0] == 0.20
    assert THRESHOLD_GRID[-1] == 0.80
    assert len(THRESHOLD_GRID) == 31
    assert all(round(right - left, 2) == 0.02 for left, right in zip(THRESHOLD_GRID, THRESHOLD_GRID[1:]))
    with pytest.raises(ValueError, match="class-collapse"):
        select_threshold([0, 1] * 10, [0.0] * 20)


def test_context_regime_metrics_and_summary_are_reported() -> None:
    record = fold_metric_record(
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [True, True, False, False],
        seed=17,
        fold=1,
    )
    assert record["context_present_macro_f1"] == 1.0
    assert record["null_context_macro_f1"] == 0.0
    summary = summarize_fold_metrics([record])
    assert summary["macro_f1"] == {
        "mean": 0.5,
        "std": 0.0,
        "minimum": 0.5,
        "maximum": 0.5,
    }


def test_candidate_selection_never_accepts_collapsed_leader() -> None:
    collapsed = CandidateEvaluation(CANDIDATE_ORDER[0], 0.99, 0.01, 0.99, 0.0, 0.95)
    healthy = CandidateEvaluation(CANDIDATE_ORDER[1], 0.70, 0.02, 0.72, 0.68, 0.60)
    assert choose_candidate([collapsed, healthy]) == healthy


def test_route_thresholds_are_diagnostic_only_and_noncollapsed() -> None:
    truth = [0, 0, 1, 1, 0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.60, 0.90, 0.20, 0.42, 0.58, 0.80]
    result = route_threshold_diagnostics(
        truth,
        probabilities,
        [True, True, True, True, False, False, False, False],
    )
    assert result["role"] == "diagnostic_only_not_selected_or_deployed"
    assert set(result["routes"]) == {"context_present", "context_absent"}
    for route in result["routes"].values():
        assert route["row_count"] == 4
        assert route["threshold"] in THRESHOLD_GRID
        assert route["metrics"]["maximum_predicted_class_share"] <= 0.90
