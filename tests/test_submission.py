import numpy as np
import pandas as pd
import pytest

from olikbochon.submission import (
    SubmissionValidationError,
    build_submission,
    validate_submission,
)


def test_submission_exact_columns_and_integer_labels() -> None:
    test_ids = pd.Series([10, 11, 12], name="id")
    sample = pd.DataFrame({"id": test_ids, "label": [1, 1, 1]})
    submission = build_submission(test_ids, np.array([0, 1, 0], dtype=np.int64), sample)
    assert list(submission.columns) == ["id", "label"]
    assert submission["id"].tolist() == [10, 11, 12]
    assert submission["label"].tolist() == [0, 1, 0]
    assert pd.api.types.is_integer_dtype(submission["label"])


@pytest.mark.parametrize("labels", [[0, 2], [0.0, 1.0], [0, None]])
def test_invalid_labels_are_rejected(labels: list[object]) -> None:
    with pytest.raises(SubmissionValidationError):
        build_submission(pd.Series([1, 2]), labels)


def test_row_count_and_id_order_mismatches_are_rejected() -> None:
    with pytest.raises(SubmissionValidationError, match="row count"):
        build_submission(pd.Series([1, 2, 3]), np.array([0, 1], dtype=np.int64))

    submission = pd.DataFrame({"id": [2, 1], "label": [0, 1]})
    with pytest.raises(SubmissionValidationError, match="ID order"):
        validate_submission(submission, pd.Series([1, 2]))


def test_sample_id_mismatch_is_rejected() -> None:
    test_ids = pd.Series([1, 2])
    sample = pd.DataFrame({"id": [2, 1], "label": [1, 1]})
    with pytest.raises(SubmissionValidationError, match="Sample submission ID order"):
        build_submission(test_ids, np.array([0, 1], dtype=np.int64), sample)


def test_duplicate_ids_are_preserved_when_the_test_contains_them() -> None:
    test_ids = pd.Series([1, 1, 2])
    submission = build_submission(test_ids, np.array([0, 1, 0], dtype=np.int64))
    assert submission["id"].tolist() == [1, 1, 2]
