from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olikbochon.v5_sparse import (
    CANDIDATES,
    CHAR_SPEC,
    LOGISTIC_CONFIG,
    PROMPT_WORD_SPEC,
    RESPONSE_CHAR_SPEC,
    WORD_SPEC,
    SparseNullModel,
    apply_routed_predictions,
    field_marked_text,
)


def absent_frame(size: int = 12, *, validation_token: str = "") -> pd.DataFrame:
    prompts = [f"shared prompt topic {index % 3} {validation_token}" for index in range(size)]
    responses = [
        f"shared answer {'faithful' if index % 2 else 'hallucinated'} {validation_token}"
        for index in range(size)
    ]
    return pd.DataFrame(
        {
            "context": ["[NULL]"] * size,
            "prompt_bn": prompts,
            "response_bn": responses,
            "label": [index % 2 for index in range(size)],
        }
    )


def test_frozen_vectorizer_and_classifier_configurations_are_exact() -> None:
    assert CHAR_SPEC.__dict__ == {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "lowercase": False,
        "sublinear_tf": True,
        "min_df": 2,
        "max_features": 30000,
        "norm": "l2",
    }
    assert WORD_SPEC.max_features == 15000
    assert PROMPT_WORD_SPEC.max_features == 8000
    assert RESPONSE_CHAR_SPEC.max_features == 20000
    assert LOGISTIC_CONFIG == {
        "class_weight": "balanced",
        "C": 1.0,
        "max_iter": 3000,
        "solver": "liblinear",
        "random_state": 42,
    }


def test_field_marked_view_reuses_v5_normalization_and_preserves_punctuation() -> None:
    assert field_marked_text("  WHAT—২০২৬? ", " উত্তর। ") == (
        "[QUESTION] what-2026?\n[ANSWER] উত্তর."
    )


def test_validation_vocabulary_never_participates_in_fit() -> None:
    training = absent_frame()
    validation = absent_frame(4, validation_token="validationonlytoken")
    model = SparseNullModel(CANDIDATES[1]).fit(training)
    vocabulary = model.vectorizers["combined_word"].vocabulary_
    assert "validationonlytoken" not in vocabulary
    model.transform(validation)
    assert "validationonlytoken" not in vocabulary
    assert model.fit_audit()["training_row_count"] == len(training)
    assert model.fit_audit()["vectorizer_fit_scope"].endswith("training_fold_only")


def test_candidate_i_fits_numeric_state_on_training_rows_only() -> None:
    training = absent_frame()
    validation = absent_frame(4, validation_token="unseenrarevalidationtoken")
    model = SparseNullModel(CANDIDATES[4]).fit(training)
    assert model.numeric_extractor is not None
    assert "unseenrarevalidationtoken" not in model.numeric_extractor.response_token_frequency
    transformed = model.transform(validation)
    assert np.isfinite(transformed.data).all()
    audit = model.fit_audit()
    assert audit["numeric_feature_count"] == 21
    assert audit["numeric_fit_scope"].endswith("training_fold_only")


def test_sparse_model_rejects_context_present_training_rows() -> None:
    frame = absent_frame()
    frame.loc[0, "context"] = "present evidence"
    with pytest.raises(ValueError, match="context-absent rows only"):
        SparseNullModel(CANDIDATES[0]).fit(frame)


def test_every_sparse_candidate_is_finite_and_deterministic() -> None:
    training = absent_frame()
    validation = absent_frame(4, validation_token="heldouttoken")
    for candidate in CANDIDATES:
        first = SparseNullModel(candidate).fit(training)
        second = SparseNullModel(candidate).fit(training)
        first_matrix = first.transform(validation)
        assert np.isfinite(first_matrix.data).all()
        assert np.array_equal(
            first.predict_label1_probability(validation),
            second.predict_label1_probability(validation),
        )


def test_present_route_is_always_the_frozen_substring_rule() -> None:
    frame = pd.DataFrame(
        {
            "context": ["ঢাকা ২০২৬", "[NULL]", None],
            "prompt_bn": ["p1", "p2", "p3"],
            "response_bn": ["ঢাকা 2026", "absent one", "absent two"],
            "label": [0, 0, 1],
        }
    )
    predictions = apply_routed_predictions(frame, [0.9, 0.1], null_threshold=0.5)
    assert predictions.tolist() == [1, 1, 0]
