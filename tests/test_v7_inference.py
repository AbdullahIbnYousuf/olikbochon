from __future__ import annotations

import pandas as pd
import pytest

from inference.run_v7_inference import TEST_COLUMNS, load_raw_test


def test_raw_test_loader_detects_schema_and_preserves_ids(tmp_path) -> None:
    path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "response_bn": ["r1", "r2"],
            "id": ["b", "a"],
            "prompt_bn": ["p1", "p2"],
            "context": ["[NULL]", "evidence"],
        }
    ).to_csv(path, index=False)
    frame = load_raw_test(path)
    assert tuple(frame.columns) == TEST_COLUMNS
    assert frame["id"].tolist() == ["b", "a"]


def test_raw_test_loader_rejects_duplicate_ids_and_extra_columns(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "id": ["a", "a"],
            "context": [None, None],
            "prompt_bn": ["p1", "p2"],
            "response_bn": ["r1", "r2"],
        }
    ).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_raw_test(duplicate)
    extra = tmp_path / "extra.csv"
    pd.DataFrame(
        {
            "id": ["a"],
            "context": [None],
            "prompt_bn": ["p"],
            "response_bn": ["r"],
            "label": [1],
        }
    ).to_csv(extra, index=False)
    with pytest.raises(ValueError, match="exactly"):
        load_raw_test(extra)
