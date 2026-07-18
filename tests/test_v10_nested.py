from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olikbochon.v10_ensemble import (
    BLEND_WEIGHTS,
    CANDIDATE_I,
    CANDIDATE_R,
    CANDIDATE_U,
    CANDIDATE_V,
    CANDIDATE_W,
    _fold_task,
    feature_configuration_fingerprint,
    select_nested_weight,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context": ["[NULL]"] * 18,
            "prompt_bn": [f"একই প্রশ্ন সাধারণ শব্দ {index % 3}" for index in range(18)],
            "response_bn": [f"একই উত্তর সাধারণ শব্দ {index % 4}" for index in range(18)],
            "label": [index % 2 for index in range(18)],
        }
    )


def test_nested_weight_uses_only_the_frozen_grid() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    candidate_i = np.asarray([0.1, 0.2, 0.8, 0.9])
    candidate_r = np.asarray([0.9, 0.8, 0.2, 0.1])
    result = select_nested_weight(truth, candidate_i, candidate_r)
    assert tuple(record["candidate_i_weight"] for record in result["grid_metrics"]) == BLEND_WEIGHTS
    assert result["candidate_i_weight"] == 0.75
    assert result["outer_validation_rows_used"] == 0


def test_nested_outer_validation_labels_do_not_change_fitted_state(tmp_path: Path) -> None:
    frame = _frame()
    matrix = np.column_stack(
        [np.linspace(0.0, 1.0, len(frame)) + offset for offset in range(14)]
    )
    payload = {
        "seed": 17,
        "fold": 1,
        "null_frame": frame,
        "candidate_r_matrix": matrix,
        "null_group_ids": np.asarray([f"group-{index}" for index in range(len(frame))]),
        "official_absent_indices": np.arange(len(frame), dtype=np.int64),
        "train_positions": np.arange(12, dtype=np.int64),
        "validation_positions": np.arange(12, 18, dtype=np.int64),
        "training_index_sha256": "a" * 64,
        "validation_index_sha256": "b" * 64,
        "feature_configuration_sha256": feature_configuration_fingerprint(),
        "fold_directory": str(tmp_path / "first"),
    }
    first = _fold_task(payload)
    changed = frame.copy()
    changed.loc[12:, "label"] = 1 - changed.loc[12:, "label"]
    second_payload = {**payload, "null_frame": changed, "fold_directory": str(tmp_path / "second")}
    second = _fold_task(second_payload)
    assert first["candidate_v_selection"] == second["candidate_v_selection"]
    assert first["candidate_w"] == second["candidate_w"]
    assert first["probabilities"] == second["probabilities"]
    assert first["inner_oof"]["complete_coverage"] is True
    assert first["inner_oof"]["outer_validation_rows_used"] == 0


def test_frozen_candidates_and_configuration_are_exact() -> None:
    assert CANDIDATE_I == "candidate_i_sparse_lexical_union"
    assert CANDIDATE_R == "candidate_r_mdeberta_semantic_verifier"
    assert (CANDIDATE_U, CANDIDATE_V, CANDIDATE_W) == (
        "candidate_u_fixed_equal_average",
        "candidate_v_nested_convex_blend",
        "candidate_w_nested_logistic_stacker",
    )
    assert len(feature_configuration_fingerprint()) == 64
