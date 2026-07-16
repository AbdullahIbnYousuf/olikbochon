from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from olikbochon.v2_data import (
    ExactAudit,
    FingerprintedFrame,
    LabelConflictError,
    add_fingerprints,
    audit_exact_partitions,
    deduplicate_by_precedence,
    final_unique_labeled_frame,
    labeled_row_fingerprint,
    text_fingerprint,
)


def frame(rows: list[tuple[object, object, object, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["context", "prompt_bn", "response_bn", "label"])


def partition(
    root: Path, name: str, rows: list[tuple[object, object, object, int]]
) -> FingerprintedFrame:
    path = root / name
    path.write_text("synthetic", encoding="utf-8")
    return FingerprintedFrame(name, path, add_fingerprints(frame(rows)))


def test_two_fingerprints_have_distinct_label_semantics() -> None:
    text0 = text_fingerprint(" প্রশ্ন ", "[NULL]", " উত্তর ")
    text1 = text_fingerprint("প্রশ্ন", None, "উত্তর")
    assert text0 == text1
    assert labeled_row_fingerprint("প্রশ্ন", None, "উত্তর", 0) != labeled_row_fingerprint(
        "প্রশ্ন", None, "উত্তর", 1
    )


def test_conflicting_labels_are_detected_by_text_fingerprint(tmp_path: Path) -> None:
    validation_rows = [("", "p", "r", 1), ("", "v", "b", 0)]
    train_rows = [("", "p", "r", 0), ("", "x", "a", 1)]
    partitions = {
        "public_train": partition(tmp_path, "train", train_rows),
        "public_validation": partition(tmp_path, "validation", validation_rows),
        "public_aggregate": partition(
            tmp_path, "aggregate", train_rows + validation_rows
        ),
        "official": partition(
            tmp_path, "official", [("", "o0", "z", 0), ("", "o1", "z", 1)]
        ),
    }
    with pytest.raises(LabelConflictError, match="conflicting text fingerprints"):
        audit_exact_partitions(partitions)


def test_split_composition_and_exact_overlap_counts(tmp_path: Path) -> None:
    train_rows = [("", "t0", "a", 0), ("", "t1", "b", 1)]
    validation_rows = [("", "v0", "c", 0), ("", "v1", "d", 1)]
    partitions = {
        "public_train": partition(tmp_path, "train", train_rows),
        "public_validation": partition(tmp_path, "validation", validation_rows),
        "public_aggregate": partition(tmp_path, "aggregate", train_rows + validation_rows),
        "official": partition(
            tmp_path, "official", [("", "o0", "e", 0), ("", "o1", "f", 1)]
        ),
    }
    audit = audit_exact_partitions(partitions)
    assert isinstance(audit, ExactAudit)
    assert audit.aggregate_composes_split is True
    assert audit.conflict_text_count == 0
    assert set(audit.text_overlap_counts.values()) == {0}


def test_holdout_precedence_and_final_combined_deduplication() -> None:
    train = add_fingerprints(
        frame([("", "shared", "x", 0), ("", "train", "x", 1)])
    )
    validation = add_fingerprints(
        frame([("", "shared", "x", 0), ("", "validation", "x", 1)])
    )
    official = add_fingerprints(frame([("", "official", "x", 0), ("", "other", "x", 1)]))
    train_clean, validation_clean, official_clean, removals = deduplicate_by_precedence(
        train, validation, official
    )
    assert removals["public_train_exact"] == 1
    assert len(train_clean) == 1
    assert len(validation_clean) == 2
    combined = final_unique_labeled_frame(train_clean, validation_clean, official_clean)
    assert len(combined) == 5
    assert list(combined.columns) == ["context", "prompt_bn", "response_bn", "label"]
