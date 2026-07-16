"""Fixed Version 2 training, evaluation, final refit, and submission variants."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from olikbochon.metrics import (
    MACRO_F1_OOF,
    classification_metrics,
    prediction_collapse_warning,
    predictions_from_label1,
    select_threshold,
    subgroup_metrics,
    threshold_table,
)
from olikbochon.modeling import build_model, label_probability
from olikbochon.preprocessing import build_text_series, context_presence_series
from olikbochon.submission import build_submission, validate_submission
from olikbochon.v2_data import final_unique_labeled_frame


V2_DEFAULT_SUBMISSION = "submission.csv"
V2_FIXED_SUBMISSION = "submission_fixed_050.csv"


@dataclass(frozen=True)
class V2EvaluationResult:
    """Aggregate-only Version 2 development and independent evaluation report."""

    selected_threshold: float
    public_validation_fixed: dict[str, Any]
    public_validation_selected_tuning: dict[str, Any]
    official_fixed: dict[str, Any]
    official_selected: dict[str, Any]
    official_fixed_context: dict[str, dict[str, Any]]
    official_selected_context: dict[str, dict[str, Any]]
    prediction_distributions: dict[str, dict[int, int]]
    collapse_warnings: tuple[str, ...]
    threshold_table: tuple[dict[str, float], ...]


def prediction_distribution(predictions: Sequence[int]) -> dict[int, int]:
    """Return aggregate prediction counts with explicit binary keys."""
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=2)
    return {0: int(counts[0]), 1: int(counts[1])}


def evaluate_v2(
    public_train: pd.DataFrame,
    public_validation: pd.DataFrame,
    official: pd.DataFrame,
) -> tuple[V2EvaluationResult, Pipeline]:
    """Fit on public train, tune on public validation, and evaluate official once."""
    model = build_model()
    train_text = build_text_series(public_train)
    train_labels = public_train["label"].to_numpy(dtype=np.int64)
    model.fit(train_text, train_labels)

    validation_text = build_text_series(public_validation)
    validation_truth = public_validation["label"].to_numpy(dtype=np.int64)
    validation_probabilities = label_probability(model, validation_text, label=1)
    table = threshold_table(validation_truth, validation_probabilities)
    selected_threshold = select_threshold(table, strategy=MACRO_F1_OOF)
    validation_fixed_predictions = predictions_from_label1(validation_probabilities, 0.5)
    validation_selected_predictions = predictions_from_label1(
        validation_probabilities, selected_threshold
    )

    official_text = build_text_series(official)
    official_truth = official["label"].to_numpy(dtype=np.int64)
    official_probabilities = label_probability(model, official_text, label=1)
    official_fixed_predictions = predictions_from_label1(official_probabilities, 0.5)
    official_selected_predictions = predictions_from_label1(
        official_probabilities, selected_threshold
    )
    official_context = context_presence_series(official).to_numpy(dtype=bool)

    prediction_sets = {
        "public_validation_fixed_050": validation_fixed_predictions,
        "public_validation_selected_tuning": validation_selected_predictions,
        "official_fixed_050": official_fixed_predictions,
        "official_selected": official_selected_predictions,
    }
    warnings = tuple(
        warning
        for name, predictions in prediction_sets.items()
        if (warning := prediction_collapse_warning(predictions, name=name)) is not None
    )

    result = V2EvaluationResult(
        selected_threshold=selected_threshold,
        public_validation_fixed=classification_metrics(
            validation_truth, validation_fixed_predictions
        ),
        public_validation_selected_tuning=classification_metrics(
            validation_truth, validation_selected_predictions
        ),
        official_fixed=classification_metrics(official_truth, official_fixed_predictions),
        official_selected=classification_metrics(
            official_truth, official_selected_predictions
        ),
        official_fixed_context=subgroup_metrics(
            official_truth, official_fixed_predictions, official_context
        ),
        official_selected_context=subgroup_metrics(
            official_truth, official_selected_predictions, official_context
        ),
        prediction_distributions={
            name: prediction_distribution(predictions)
            for name, predictions in prediction_sets.items()
        },
        collapse_warnings=warnings,
        threshold_table=tuple(
            {
                "threshold": float(row.threshold),
                "f1_label0": float(row.f1_label0),
                "f1_label1": float(row.f1_label1),
                "macro_f1": float(row.macro_f1),
                "accuracy": float(row.accuracy),
            }
            for row in table.itertuples(index=False)
        ),
    )
    return result, model


def fit_final_v2_model(
    public_train: pd.DataFrame,
    public_validation: pd.DataFrame,
    official: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame]:
    """Refit the unchanged pipeline on all unique allowed labeled rows."""
    combined = final_unique_labeled_frame(public_train, public_validation, official)
    model = build_model()
    model.fit(build_text_series(combined), combined["label"].to_numpy(dtype=np.int64))
    return model, combined


def build_v2_submission_variants(
    test_ids: pd.Series,
    probabilities_label1: Sequence[float],
    selected_threshold: float,
    sample_submission: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build and independently validate macro-selected and fixed-0.50 outputs."""
    probabilities = np.asarray(probabilities_label1, dtype=np.float64)
    variants = {
        V2_DEFAULT_SUBMISSION: build_submission(
            test_ids,
            predictions_from_label1(probabilities, selected_threshold),
            sample_submission,
        ),
        V2_FIXED_SUBMISSION: build_submission(
            test_ids,
            predictions_from_label1(probabilities, 0.5),
            sample_submission,
        ),
    }
    normalized_ids = test_ids.reset_index(drop=True)
    for submission in variants.values():
        validate_submission(submission, normalized_ids, sample_submission)
    return variants
