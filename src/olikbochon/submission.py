"""Strict submission construction and validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


TEST_COLUMNS = ("id", "context", "prompt_bn", "response_bn")
SUBMISSION_COLUMNS = ("id", "label")


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
