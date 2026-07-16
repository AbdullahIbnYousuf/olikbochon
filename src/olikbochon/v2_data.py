"""Version 2 labeled-data roles, deterministic fingerprints, and exact audits."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from olikbochon.data_loading import DataValidationError, sha256_file, validate_labeled_frame
from olikbochon.preprocessing import build_marked_text, context_presence_series


PUBLIC_AGGREGATE_FILENAME = "bangla_hallucination_5k_contrastive.json"
PUBLIC_TRAIN_FILENAME = "bangla_hallucination_5k_train.json"
PUBLIC_VALIDATION_FILENAME = "bangla_hallucination_5k_validation.json"
OFFICIAL_FILENAME = "dataset samples.json"


class V2AuditError(RuntimeError):
    """Raised when the approved Version 2 data contract is unsafe."""


class LabelConflictError(V2AuditError):
    """Raised when identical normalized text has more than one label."""


@dataclass(frozen=True)
class FingerprintedFrame:
    """One validated labeled partition with deterministic anonymous keys."""

    name: str
    path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class ExactAudit:
    """Aggregate exact-duplicate facts without raw labeled content."""

    partition_stats: dict[str, dict[str, Any]]
    text_overlap_counts: dict[str, int]
    labeled_overlap_counts: dict[str, int]
    conflict_text_count: int
    conflict_partitions: tuple[str, ...]
    aggregate_composes_split: bool


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_fingerprint(prompt: Any, context: Any, response: Any) -> str:
    """Hash normalized field-marked text without the label."""
    return _hash_text(build_marked_text(prompt, context, response))


def labeled_row_fingerprint(prompt: Any, context: Any, response: Any, label: int) -> str:
    """Hash normalized field-marked text plus an explicit binary label."""
    marked = build_marked_text(prompt, context, response)
    return _hash_text(f"{marked}\n\n__LABEL__\n{int(label)}")


def add_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with text-only and labeled-row SHA-256 keys."""
    validated = validate_labeled_frame(frame)
    result = validated.copy(deep=True)
    tuples = result[["prompt_bn", "context", "response_bn", "label"]].itertuples(
        index=False, name=None
    )
    text_keys: list[str] = []
    labeled_keys: list[str] = []
    for prompt, context, response, label in tuples:
        text_keys.append(text_fingerprint(prompt, context, response))
        labeled_keys.append(labeled_row_fingerprint(prompt, context, response, int(label)))
    result["text_fingerprint"] = text_keys
    result["labeled_row_fingerprint"] = labeled_keys
    return result


def load_v2_labeled_json(path: Path, expected_name: str) -> FingerprintedFrame:
    """Load one explicitly named labeled JSON; CSV inputs are impossible here."""
    path = Path(path)
    if path.name != expected_name:
        raise DataValidationError(f"Expected filename {expected_name!r}, found {path.name!r}")
    frame = pd.read_json(path, orient="records")
    return FingerprintedFrame(expected_name, path, add_fingerprints(frame))


def _partition_stats(item: FingerprintedFrame) -> dict[str, Any]:
    frame = item.frame
    return {
        "filename": item.path.name,
        "sha256": sha256_file(item.path),
        "rows": int(len(frame)),
        "columns": list(frame.columns[:4]),
        "label_dtype": str(frame["label"].dtype),
        "label_counts": {
            int(key): int(value)
            for key, value in frame["label"].value_counts().sort_index().items()
        },
        "missing_values": {
            column: int(frame[column].isna().sum())
            for column in ("context", "prompt_bn", "response_bn", "label")
        },
        "context_counts": {
            bool(key): int(value)
            for key, value in context_presence_series(frame).value_counts().sort_index().items()
        },
        "duplicate_text_rows": int(frame["text_fingerprint"].duplicated(keep=False).sum()),
        "duplicate_labeled_rows": int(
            frame["labeled_row_fingerprint"].duplicated(keep=False).sum()
        ),
        "id_column_present": "id" in frame.columns,
    }


def _overlap(left: pd.DataFrame, right: pd.DataFrame, column: str) -> int:
    return len(set(left[column]).intersection(right[column]))


def audit_exact_partitions(
    partitions: Mapping[str, FingerprintedFrame],
) -> ExactAudit:
    """Audit composition, duplicates, overlaps, and text-label conflicts."""
    required = {"public_train", "public_validation", "public_aggregate", "official"}
    if set(partitions) != required:
        raise V2AuditError(f"Expected partitions {sorted(required)}, found {sorted(partitions)}")

    combined_parts: list[pd.DataFrame] = []
    for name, item in partitions.items():
        part = item.frame[["text_fingerprint", "labeled_row_fingerprint", "label"]].copy()
        part["partition"] = name
        combined_parts.append(part)
    combined = pd.concat(combined_parts, ignore_index=True)

    conflict_groups = combined.groupby("text_fingerprint", sort=False)["label"].nunique()
    conflict_keys = set(conflict_groups[conflict_groups > 1].index)
    conflict_partitions = tuple(
        sorted(combined.loc[combined["text_fingerprint"].isin(conflict_keys), "partition"].unique())
    )

    train = partitions["public_train"].frame
    validation = partitions["public_validation"].frame
    aggregate = partitions["public_aggregate"].frame
    official = partitions["official"].frame
    split_counter = Counter(train["labeled_row_fingerprint"]) + Counter(
        validation["labeled_row_fingerprint"]
    )
    aggregate_counter = Counter(aggregate["labeled_row_fingerprint"])

    pairs = {
        "public_train__public_validation": (train, validation),
        "public_train__official": (train, official),
        "public_validation__official": (validation, official),
        "public_aggregate__official": (aggregate, official),
    }
    audit = ExactAudit(
        partition_stats={name: _partition_stats(item) for name, item in partitions.items()},
        text_overlap_counts={
            name: _overlap(left, right, "text_fingerprint")
            for name, (left, right) in pairs.items()
        },
        labeled_overlap_counts={
            name: _overlap(left, right, "labeled_row_fingerprint")
            for name, (left, right) in pairs.items()
        },
        conflict_text_count=len(conflict_keys),
        conflict_partitions=conflict_partitions,
        aggregate_composes_split=split_counter == aggregate_counter,
    )
    if conflict_keys:
        raise LabelConflictError(
            f"Found {len(conflict_keys)} conflicting text fingerprints across partitions "
            f"{list(conflict_partitions)}"
        )
    if not audit.aggregate_composes_split:
        raise V2AuditError("Public 4k and 1k labeled rows do not exactly compose the 5k aggregate")
    return audit


def deduplicate_by_precedence(
    public_train: pd.DataFrame,
    public_validation: pd.DataFrame,
    official: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Protect official then validation roles by removing lower-precedence exact copies."""
    official_clean = official.drop_duplicates(
        "labeled_row_fingerprint", keep="first"
    ).reset_index(drop=True)
    official_text = set(official_clean["text_fingerprint"])

    validation_without_official = public_validation.loc[
        ~public_validation["text_fingerprint"].isin(official_text)
    ]
    validation_clean = validation_without_official.drop_duplicates(
        "labeled_row_fingerprint", keep="first"
    ).reset_index(drop=True)
    protected_text = official_text.union(validation_clean["text_fingerprint"])

    train_without_holdouts = public_train.loc[
        ~public_train["text_fingerprint"].isin(protected_text)
    ]
    train_clean = train_without_holdouts.drop_duplicates(
        "labeled_row_fingerprint", keep="first"
    ).reset_index(drop=True)

    removals = {
        "public_train_exact": int(len(public_train) - len(train_clean)),
        "public_validation_exact": int(len(public_validation) - len(validation_clean)),
        "official_exact": int(len(official) - len(official_clean)),
    }
    return train_clean, validation_clean, official_clean, removals


def drop_audit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the four-column model frame without anonymous audit keys."""
    return frame[["context", "prompt_bn", "response_bn", "label"]].reset_index(drop=True)


def final_unique_labeled_frame(
    public_train: pd.DataFrame,
    public_validation: pd.DataFrame,
    official: pd.DataFrame,
) -> pd.DataFrame:
    """Combine allowed rows once, failing if text-label conflicts remain."""
    combined = pd.concat([public_train, public_validation, official], ignore_index=True)
    conflicts = combined.groupby("text_fingerprint")["label"].nunique()
    if bool((conflicts > 1).any()):
        raise LabelConflictError(
            f"Final combination contains {int((conflicts > 1).sum())} conflicting texts"
        )
    unique = combined.drop_duplicates("labeled_row_fingerprint", keep="first")
    return drop_audit_columns(unique)
