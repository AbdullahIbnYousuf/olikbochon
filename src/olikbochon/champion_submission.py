"""Locked full-data inference primitives for the official-only current champion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import validate_labeled_frame
from .metrics import predictions_from_label1
from .submission import TEST_COLUMNS, build_submission, validate_test_frame
from .v4_preprocessing import official_context_is_present
from .v5_features import CONTEXT_ABSENT_FEATURES
from .v5_lexical import deterministic_substring_prediction
from .v5_sparse import (
    CANDIDATES,
    CHAR_SPEC,
    LOGISTIC_CONFIG,
    WORD_SPEC,
    SparseNullModel,
)


CHAMPION_CANDIDATE = "candidate_i_sparse_lexical_union"
CHAMPION_THRESHOLD = 0.50
EXPECTED_TRAINING_ROWS = 299
EXPECTED_PRESENT_TRAINING_ROWS = 130
EXPECTED_ABSENT_TRAINING_ROWS = 169
EXPECTED_TEST_ROWS = 2_516
PRESENT_ROUTE = "context_present_substring_rule"
ABSENT_ROUTE = "context_absent_candidate_i"
PROBABILITY_COLUMNS = (
    "id",
    "route",
    "label_1_probability",
    "deterministic_rule_score",
    "predicted_label",
)


@dataclass(frozen=True)
class FittedChampion:
    model: SparseNullModel
    training_audit: dict[str, Any]


@dataclass(frozen=True)
class ChampionOutputs:
    submission: pd.DataFrame
    probabilities: pd.DataFrame
    inference_audit: dict[str, Any]


def frozen_champion_config() -> dict[str, Any]:
    return {
        "champion": "v5_candidate_a_plus_sparse_candidate_i",
        "training_source": "authenticated_official_labeled_sample_only",
        "present_route": "exact normalized response substring in normalized context",
        "absent_route": CHAMPION_CANDIDATE,
        "null_threshold": CHAMPION_THRESHOLD,
        "character_vectorizer": CHAR_SPEC.__dict__,
        "word_vectorizer": WORD_SPEC.__dict__,
        "numeric_features": list(CONTEXT_ABSENT_FEATURES),
        "numeric_scaler": "StandardScaler fitted on all 169 official absent rows",
        "rare_token_fit_scope": "all 169 official absent rows only",
        "classifier": LOGISTIC_CONFIG,
        "probability_convention": {
            "context_absent": "label_1_probability from Candidate I",
            "context_present": (
                "label_1_probability is empty; deterministic_rule_score is 0 or 1"
            ),
        },
    }


def fit_frozen_champion(frame: pd.DataFrame) -> FittedChampion:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    if len(validated) != EXPECTED_TRAINING_ROWS:
        raise ValueError("Champion training requires exactly 299 official rows")
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_TRAINING_ROWS:
        raise ValueError("Champion training requires exactly 130 context-present rows")
    if int((~presence).sum()) != EXPECTED_ABSENT_TRAINING_ROWS:
        raise ValueError("Champion training requires exactly 169 context-absent rows")
    if CHAMPION_CANDIDATE != CANDIDATES[4] or CHAMPION_THRESHOLD != 0.50:
        raise RuntimeError("Frozen champion candidate or threshold changed")
    absent = validated.loc[~presence].reset_index(drop=True)
    model = SparseNullModel(CHAMPION_CANDIDATE).fit(absent)
    matrix = model.transform(absent)
    fit_audit = model.fit_audit()
    if fit_audit["training_row_count"] != EXPECTED_ABSENT_TRAINING_ROWS:
        raise RuntimeError("Candidate I fitted an unexpected number of rows")
    return FittedChampion(
        model,
        {
            "official_training_rows": len(validated),
            "context_present_training_rows": int(presence.sum()),
            "context_absent_training_rows": int((~presence).sum()),
            "candidate_fit": fit_audit,
            "training_matrix_rows": int(matrix.shape[0]),
            "training_matrix_columns": int(matrix.shape[1]),
            "classifier_classes": [int(value) for value in model.classifier.classes_],
            "test_rows_used_for_fit": 0,
        },
    )


def validate_champion_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_test_frame(frame)
    if len(frame) != EXPECTED_TEST_ROWS:
        raise ValueError("Champion inference requires exactly 2,516 competition rows")
    if frame["id"].duplicated().any():
        raise ValueError("Competition test IDs must be unique")
    if frame["prompt_bn"].isna().any() or frame["response_bn"].isna().any():
        raise ValueError("Competition prompts and responses must not be missing")
    return frame.loc[:, TEST_COLUMNS].copy(deep=True).reset_index(drop=True)


def predict_frozen_champion(
    fitted: FittedChampion,
    test_frame: pd.DataFrame,
) -> ChampionOutputs:
    test = validate_champion_test_frame(test_frame)
    presence = np.asarray(
        [official_context_is_present(value) for value in test["context"]], dtype=bool
    )
    absent = test.loc[~presence].reset_index(drop=True)
    absent_probabilities = fitted.model.predict_unlabeled_label1_probability(absent)
    predictions = np.empty(len(test), dtype=np.int64)
    predictions[~presence] = predictions_from_label1(
        absent_probabilities, CHAMPION_THRESHOLD
    )
    present = test.loc[presence]
    predictions[presence] = np.asarray(
        [
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in present.itertuples(index=False)
        ],
        dtype=np.int64,
    )
    submission = build_submission(test["id"], predictions)
    label_1_probability = pd.Series(pd.NA, index=test.index, dtype="Float64")
    deterministic_score = pd.Series(pd.NA, index=test.index, dtype="Int64")
    label_1_probability.loc[~presence] = absent_probabilities
    deterministic_score.loc[presence] = predictions[presence]
    probabilities = pd.DataFrame(
        {
            "id": test["id"],
            "route": np.where(presence, PRESENT_ROUTE, ABSENT_ROUTE),
            "label_1_probability": label_1_probability,
            "deterministic_rule_score": deterministic_score,
            "predicted_label": predictions,
        }
    )
    if tuple(probabilities.columns) != PROBABILITY_COLUMNS:
        raise RuntimeError("Champion probability artifact schema changed")
    if probabilities["id"].tolist() != test["id"].tolist():
        raise RuntimeError("Champion probability artifact changed test ID order")
    if not probabilities["predicted_label"].isin([0, 1]).all():
        raise RuntimeError("Champion predictions must be binary")
    counts = np.bincount(predictions, minlength=2)
    return ChampionOutputs(
        submission,
        probabilities,
        {
            "test_rows": len(test),
            "test_route_counts": {
                "context_present": int(presence.sum()),
                "context_absent": int((~presence).sum()),
            },
            "predicted_label_counts": {"0": int(counts[0]), "1": int(counts[1])},
            "maximum_predicted_class_share": float(counts.max() / len(test)),
            "test_rows_used_for_fit": 0,
            "test_rows_printed_or_manually_reviewed": 0,
            "raw_test_text_persisted": False,
        },
    )
