from __future__ import annotations

import inspect

import joblib
import numpy as np
import pandas as pd
import pytest

from olikbochon.data_loading import DataValidationError
from olikbochon.v5_sparse import CANDIDATES, SparseNullModel
from olikbochon.v4_wikipedia import RETRIEVAL_THRESHOLDS, RetrievalResult, build_lexical_frame
from olikbochon.v5_lexical import build_v5_groups
from olikbochon.v7_e0 import (
    CLASSIFICATION_THRESHOLDS,
    effective_contexts,
    select_inner_configuration as select_e0_inner_configuration,
)
from olikbochon.v7_e1 import select_inner_configuration
from olikbochon.v7_e3 import RULES, route_rule_probability, select_rule
from olikbochon.v7_e2 import select_inner_configuration as select_e2_inner_configuration
from olikbochon.v7_e4 import select_inner_configuration as select_e4_inner_configuration
from olikbochon.v7_protocol import (
    FitScope,
    SelectionAudit,
    build_common_fold_assignments,
    outer_indices,
    stable_training_row_ids,
)
from test_v7_group_folds import synthetic_frame


def absent_frame(prefix: str, rows: int) -> pd.DataFrame:
    records = []
    for index in range(rows):
        repeated = "sharedtoken sharedtoken" if index < rows else ""
        records.append(
            {
                "context": "[NULL]",
                "prompt_bn": f"{prefix} prompt {index} {repeated}",
                "response_bn": f"{prefix} response {index} sharedanswer",
                "label": index % 2,
            }
        )
    return pd.DataFrame(records, columns=["context", "prompt_bn", "response_bn", "label"])


def test_fit_and_selection_scopes_reject_outer_or_test_leakage() -> None:
    FitScope("fit", ("a", "b"), ("c",)).validate()
    with pytest.raises(DataValidationError, match="leaks"):
        FitScope("fit", ("a", "b"), ("b", "c")).validate()
    with pytest.raises(DataValidationError, match="test rows"):
        FitScope("fit", ("a",), ("b",), test_row_count=1).validate()
    SelectionAudit("classification_threshold", ("a",), ("b",)).validate()
    with pytest.raises(DataValidationError, match="outer-validation"):
        SelectionAudit(
            "retrieval_acceptance_threshold", ("a",), ("b",), ("outer",)
        ).validate()
    with pytest.raises(DataValidationError, match="test rows"):
        SelectionAudit("fusion_regularization", ("a",), ("b",), test_row_count=1).validate()


def test_candidate_i_vectorizers_fit_training_rows_only_and_reload_equal(tmp_path) -> None:
    train = absent_frame("training", 12)
    validation = absent_frame("validation_secret_xyz", 4)
    model = SparseNullModel(CANDIDATES[4], classifier_c=1.0).fit(train)
    audit = model.fit_audit()
    assert audit["training_row_count"] == len(train)
    assert audit["vectorizer_fit_scope"].endswith("training_fold_only")
    assert audit["numeric_fit_scope"].endswith("training_fold_only")
    assert "validation_secret_xyz" not in model.vectorizers["combined_word"].vocabulary_
    probabilities = model.predict_label1_probability(validation)
    path = tmp_path / "candidate_i.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    assert np.allclose(
        probabilities,
        reloaded.predict_label1_probability(validation),
        rtol=0.0,
        atol=1e-12,
    )


def test_e1_has_no_competition_test_loader_or_outer_threshold_argument() -> None:
    source = inspect.getsource(select_inner_configuration)
    assert "outer_validation" not in inspect.signature(select_inner_configuration).parameters
    assert "make_inner_grouped_folds" in source
    assert "test set.csv" not in source.lower()


def test_e0_retrieval_and_classification_selection_is_inner_only() -> None:
    frame = synthetic_frame()
    labels = frame["label"].to_numpy(dtype=np.int64)
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    groups = np.asarray(build_v5_groups(frame).group_ids, dtype=object)
    assignments = build_common_fold_assignments(frame)
    outer_train, _ = outer_indices(assignments, row_ids, seed=17, outer_fold=1)
    retrieval = RetrievalResult(
        tuple(f"retrieved-evidence-{index}" for index in range(len(frame))),
        np.linspace(0.05, 0.40, len(frame), dtype=np.float64),
    )
    features = {
        threshold: build_lexical_frame(
            effective_contexts(frame, retrieval, threshold), frame["response_bn"]
        )
        for threshold in RETRIEVAL_THRESHOLDS
    }
    retrieval_threshold, classification_threshold, audit, probabilities = (
        select_e0_inner_configuration(
            frame,
            labels,
            row_ids,
            groups,
            outer_train,
            features,
            seed=17,
            outer_fold=1,
        )
    )
    assert retrieval_threshold in RETRIEVAL_THRESHOLDS
    assert classification_threshold in CLASSIFICATION_THRESHOLDS
    assert audit["selection_scope"] == "outer_training_inner_grouped_oof_only"
    assert probabilities.shape == (len(outer_train),)
    assert np.isfinite(probabilities).all()


def test_e3_search_space_is_small_deterministic_and_route_aware() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    present = np.asarray([True, True, True, True, False, False, False, False])
    e0 = np.asarray([0.1, 0.9, 0.2, 0.8, 0.2, 0.8, 0.3, 0.7])
    e1 = np.asarray([0.0, 1.0, 0.0, 1.0, 0.4, 0.6, 0.45, 0.55])
    assert len(RULES) == 5
    blended = route_rule_probability(
        "e1_present_equal_absent_blend", e0, e1, present
    )
    assert np.array_equal(blended[present], e1[present])
    assert np.allclose(blended[~present], 0.5 * e0[~present] + 0.5 * e1[~present])
    rule, threshold, audit = select_rule(labels, e0, e1, present)
    assert rule in RULES
    assert 0.40 <= threshold <= 0.60
    assert audit["selection_scope"].endswith("cross_fitted_bases_only")


def test_e2_and_e4_selection_interfaces_cannot_receive_outer_validation() -> None:
    for selector in (select_e2_inner_configuration, select_e4_inner_configuration):
        signature = inspect.signature(selector)
        source = inspect.getsource(selector)
        assert "outer_validation" not in signature.parameters
        assert "make_inner_grouped_folds" in source
        assert "test set.csv" not in source.lower()
