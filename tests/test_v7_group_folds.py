from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from olikbochon.v5_lexical import build_v5_groups
from olikbochon.v7_protocol import (
    OUTER_SEEDS,
    build_common_fold_assignments,
    make_inner_grouped_folds,
    outer_indices,
    stable_training_row_ids,
)


def synthetic_frame(rows: int = 50) -> pd.DataFrame:
    records = []
    for index in range(rows):
        token = hashlib.sha256(f"v7-row-{index}".encode()).hexdigest()
        records.append(
            {
                "context": f"evidence-{token}" if index % 3 else "[NULL]",
                "prompt_bn": f"question-{token}",
                "response_bn": f"answer-{token}",
                "label": index % 2,
            }
        )
    return pd.DataFrame(records, columns=["context", "prompt_bn", "response_bn", "label"])


def test_common_outer_folds_have_no_group_overlap_and_complete_coverage() -> None:
    frame = synthetic_frame()
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    groups = np.asarray(build_v5_groups(frame).group_ids, dtype=object)
    assignments = build_common_fold_assignments(frame)
    for seed in OUTER_SEEDS:
        seed_rows = assignments.loc[assignments["seed"] == seed]
        assert not seed_rows["row_id"].duplicated().any()
        assert set(seed_rows["row_id"]) == set(row_ids)
        for outer_fold in range(1, 6):
            train, validation = outer_indices(
                assignments, row_ids, seed=seed, outer_fold=outer_fold
            )
            assert not (set(groups[train]) & set(groups[validation]))
            assert set(frame.iloc[validation]["label"]) == {0, 1}


def test_inner_folds_use_only_outer_training_and_keep_groups_isolated() -> None:
    frame = synthetic_frame()
    labels = frame["label"].to_numpy(dtype=np.int64)
    groups = np.asarray(build_v5_groups(frame).group_ids, dtype=object)
    row_ids = stable_training_row_ids(frame)
    assignments = build_common_fold_assignments(frame)
    outer_train, outer_validation = outer_indices(
        assignments, row_ids, seed=17, outer_fold=1
    )
    splits = make_inner_grouped_folds(
        labels, groups, outer_train, seed=17, outer_fold=1
    )
    coverage: list[int] = []
    for split in splits:
        train = np.asarray(split.train_indices)
        validation = np.asarray(split.validation_indices)
        assert set(train).issubset(set(outer_train))
        assert set(validation).issubset(set(outer_train))
        assert not (set(validation) & set(outer_validation))
        assert not (set(groups[train]) & set(groups[validation]))
        coverage.extend(validation.tolist())
    assert sorted(coverage) == sorted(outer_train.tolist())
