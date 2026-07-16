"""Strict labeled-data loading that never discovers or reads competition test text."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELED_FILENAME = "dataset samples.json"
LABELED_COLUMNS = ("context", "prompt_bn", "response_bn", "label")
OFFICIAL_SAMPLE_SHA256 = "f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28"


class DataValidationError(ValueError):
    """Raised when a labeled input violates the Version 1 data contract."""


def sha256_file(path: Path) -> str:
    """Hash a file as metadata without decoding or displaying its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy the official labeled schema without printing examples."""
    if tuple(frame.columns) != LABELED_COLUMNS:
        raise DataValidationError(
            f"Expected columns {list(LABELED_COLUMNS)}, found {list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError("Labeled dataset must be nonempty")
    if frame["label"].isna().any():
        raise DataValidationError("Labels must not be missing")

    labels = frame["label"].tolist()
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) not in {0, 1}
        for value in labels
    ):
        raise DataValidationError("Labels must be integers restricted to 0 and 1")
    if set(map(int, labels)) != {0, 1}:
        raise DataValidationError("Both label classes 0 and 1 must be present")
    if frame["prompt_bn"].isna().any() or frame["response_bn"].isna().any():
        raise DataValidationError("Prompt and response values must not be missing")

    validated = frame.copy(deep=True)
    validated["label"] = validated["label"].astype(np.int64)
    return validated


def load_labeled_json(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    """Load one explicitly supplied official labeled JSON file."""
    path = Path(path)
    if path.name != LABELED_FILENAME:
        raise DataValidationError(f"Expected filename {LABELED_FILENAME!r}, found {path.name!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None:
        observed_hash = sha256_file(path)
        if observed_hash != expected_sha256:
            raise DataValidationError(
                f"Official labeled file hash mismatch: expected {expected_sha256}, "
                f"observed {observed_hash}"
            )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise DataValidationError("Labeled JSON must contain a top-level list of records")
    return validate_labeled_frame(pd.DataFrame(payload))
