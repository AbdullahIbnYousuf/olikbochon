from __future__ import annotations

import numpy as np
import pandas as pd

from olikbochon.data_loading import DataValidationError
from olikbochon.v7_protocol import (
    build_common_fold_assignments,
    outer_indices,
    stable_training_row_ids,
)
from test_v7_group_folds import synthetic_frame


def test_fold_resolution_aligns_by_stable_id_after_dataframe_shuffle() -> None:
    original = synthetic_frame()
    assignments = build_common_fold_assignments(original)
    original_ids = stable_training_row_ids(original)
    _, original_validation = outer_indices(
        assignments, original_ids, seed=29, outer_fold=3
    )
    expected_ids = {original_ids[index] for index in original_validation}

    shuffled = original.sample(frac=1.0, random_state=812).reset_index(drop=True)
    shuffled_ids = stable_training_row_ids(shuffled)
    _, shuffled_validation = outer_indices(
        assignments, shuffled_ids, seed=29, outer_fold=3
    )
    assert {shuffled_ids[index] for index in shuffled_validation} == expected_ids


def test_content_derived_ids_reject_ambiguous_duplicate_rows() -> None:
    frame = synthetic_frame(10)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    try:
        stable_training_row_ids(duplicated)
    except DataValidationError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("ambiguous row IDs must be rejected")
