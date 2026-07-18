from __future__ import annotations

import numpy as np
import pandas as pd

from olikbochon.v5_features import (
    CONTEXT_ABSENT_FEATURES,
    CONTEXT_PRESENT_FEATURES,
    V5FeatureExtractor,
    aggregate_feature_audit,
)


def frame_for_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context": ["ঢাকা ২০২৬ সালে এগোয়।", "[NULL]", "ঢাকা ২০২৬ সালে এগোয়।", None],
            "prompt_bn": ["কোথায়?", "কে?", "কোথায়?", "কে?"],
            "response_bn": ["ঢাকা 2026", "সম্ভবত না, OpenAI ২০২৬", "অন্য কথা", "সঠিক"],
            "label": [1, 0, 0, 1],
        }
    )


def test_context_present_alignment_features_are_interpretable() -> None:
    frame = frame_for_features()
    extractor = V5FeatureExtractor.fit(frame)
    row = extractor.context_present_features(
        frame.iloc[0].prompt_bn,
        frame.iloc[0].context,
        frame.iloc[0].response_bn,
    )
    assert tuple(row) == CONTEXT_PRESENT_FEATURES
    assert row["response_in_context_no_punctuation"] == 1.0
    assert row["response_token_coverage"] == 1.0
    assert row["numeric_consistency_ratio"] == 1.0
    assert row["numeral_mismatch_count"] == 0.0


def test_context_absent_features_include_markers_scripts_and_specificity() -> None:
    frame = frame_for_features()
    extractor = V5FeatureExtractor.fit(frame)
    row = extractor.context_absent_features(frame.iloc[1].prompt_bn, frame.iloc[1].response_bn)
    assert tuple(row) == CONTEXT_ABSENT_FEATURES
    assert row["response_negation_count"] == 1.0
    assert row["response_uncertainty_count"] == 1.0
    assert row["response_numeral_count"] == 4.0
    assert row["prompt_has_question_mark"] == 1.0
    assert row["response_specificity_score"] >= row["response_numeral_count"]


def test_rare_token_state_is_fitted_only_from_the_supplied_training_frame() -> None:
    training = pd.DataFrame(
        {
            "response_bn": ["সাধারণ সাধারণ", "দুর্লভ"],
        }
    )
    extractor = V5FeatureExtractor.fit(training)
    assert extractor.response_token_frequency == {"সাধারণ": 2, "দুর্লভ": 1}
    assert extractor.context_absent_features("প্রশ্ন", "দুর্লভ নতুন")[
        "response_rare_token_count"
    ] == 2.0


def test_aggregate_audit_contains_only_summaries_and_no_row_values() -> None:
    audit = aggregate_feature_audit(frame_for_features())
    assert audit["row_count"] == 4
    assert audit["context_present_count"] == 2
    assert audit["context_absent_count"] == 2
    assert audit["row_level_values_persisted"] is False
    summary = audit["routes"]["context_present"]["label_1"]["features"][
        "response_token_coverage"
    ]
    assert set(summary) == {"count", "mean", "median", "std", "minimum", "maximum"}


def test_feature_matrices_are_finite_and_follow_the_frozen_order() -> None:
    frame = frame_for_features()
    extractor = V5FeatureExtractor.fit(frame)
    present = extractor.matrix(frame, [0, 2], context_present=True)
    absent = extractor.matrix(frame, [1, 3], context_present=False)
    assert present.shape == (2, len(CONTEXT_PRESENT_FEATURES))
    assert absent.shape == (2, len(CONTEXT_ABSENT_FEATURES))
    assert np.isfinite(present).all()
    assert np.isfinite(absent).all()
