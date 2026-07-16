"""Strict submission construction and validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from olikbochon.metrics import (
    CLASS0_F1_OOF_EXPERIMENTAL,
    FIXED_050,
    MACRO_F1_OOF,
)


TEST_COLUMNS = ("id", "context", "prompt_bn", "response_bn")
SUBMISSION_COLUMNS = ("id", "label")
SUBMISSION_FILENAMES = {
    MACRO_F1_OOF: "submission.csv",
    FIXED_050: "submission_fixed_050.csv",
    CLASS0_F1_OOF_EXPERIMENTAL: "submission_class0_experimental.csv",
}


class SubmissionValidationError(ValueError):
    """Raised when test metadata or a submission violates the required contract."""


def validate_test_frame(frame: pd.DataFrame) -> None:
    """Validate test structure without displaying or semantically inspecting text."""
    if tuple(frame.columns) != TEST_COLUMNS:
        raise SubmissionValidationError(
            f"Expected test columns {list(TEST_COLUMNS)}, found {list(frame.columns)}"
        )
    if frame["id"].isna().any():
        raise SubmissionValidationError("Test IDs must not be missing")


def _integer_binary_labels(labels: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(labels.dtype):
        return False
    return bool(labels.isin([0, 1]).all())


def validate_submission(
    submission: pd.DataFrame,
    test_ids: pd.Series,
    sample_submission: pd.DataFrame | None = None,
) -> None:
    """Validate exact columns, ID sequence, row count, and integer binary labels."""
    if tuple(submission.columns) != SUBMISSION_COLUMNS:
        raise SubmissionValidationError(
            f"Submission columns must be exactly {list(SUBMISSION_COLUMNS)}"
        )
    if len(submission) != len(test_ids):
        raise SubmissionValidationError("Submission row count must match test row count")
    if submission["id"].isna().any():
        raise SubmissionValidationError("Submission IDs must not be missing")
    if submission["label"].isna().any():
        raise SubmissionValidationError("Submission labels must not be missing")
    if submission["id"].tolist() != test_ids.tolist():
        raise SubmissionValidationError("Submission IDs must preserve test ID order")
    if not _integer_binary_labels(submission["label"]):
        raise SubmissionValidationError("Submission labels must be integer values 0 or 1")

    if sample_submission is not None:
        if tuple(sample_submission.columns) != SUBMISSION_COLUMNS:
            raise SubmissionValidationError("Sample submission must have exact id,label columns")
        if len(sample_submission) != len(test_ids):
            raise SubmissionValidationError("Sample submission row count must match test data")
        if sample_submission["id"].tolist() != test_ids.tolist():
            raise SubmissionValidationError("Sample submission ID order does not match test IDs")


def build_submission(
    test_ids: pd.Series,
    predicted_labels: Sequence[int],
    sample_submission: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a validated submission while preserving test IDs exactly."""
    labels_array = np.asarray(predicted_labels)
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise SubmissionValidationError("Predicted labels must have an integer dtype")
    if len(labels_array) != len(test_ids):
        raise SubmissionValidationError("Submission row count must match test row count")
    submission = pd.DataFrame(
        {"id": test_ids.reset_index(drop=True), "label": labels_array.astype(np.int64)}
    )
    validate_submission(submission, test_ids.reset_index(drop=True), sample_submission)
    return submission


def build_submission_variants(
    test_ids: pd.Series,
    predictions_by_strategy: Mapping[str, Sequence[int]],
    sample_submission: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build and independently validate every named Version 1 submission variant."""
    if set(predictions_by_strategy) != set(SUBMISSION_FILENAMES):
        raise SubmissionValidationError(
            f"Submission strategies must be exactly {sorted(SUBMISSION_FILENAMES)}"
        )
    variants: dict[str, pd.DataFrame] = {}
    normalized_ids = test_ids.reset_index(drop=True)
    for strategy in SUBMISSION_FILENAMES:
        submission = build_submission(
            normalized_ids,
            predictions_by_strategy[strategy],
            sample_submission,
        )
        validate_submission(submission, normalized_ids, sample_submission)
        variants[strategy] = submission
    return variants
