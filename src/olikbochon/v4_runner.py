"""Non-training Version 4 preparation skeleton for official labeled data only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json
from .v3_preprocessing import raw_context_is_present
from .v4_preprocessing import (
    EncodedV4Input,
    encode_comparison_baseline,
    encode_field_aware,
    prepare_v4_input,
    truncation_statistics,
)
from .v4_training import V4TrainingConfig
from .v4_validation import RepeatedGroupedFolds, build_repeated_official_folds


@dataclass(frozen=True)
class PreparedExperiment:
    """Aggregate-safe prepared inputs and frozen folds; no training state."""

    encodings: tuple[EncodedV4Input, ...]
    folds: RepeatedGroupedFolds
    truncation_summary: dict[str, Any]


def official_training_path(repository_root: Path) -> Path:
    """Resolve only the allowlisted official labeled sample path."""
    path = Path(repository_root).resolve() / "data" / "competition" / "dataset samples.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_official_training_frame(repository_root: Path) -> Any:
    """Load the authenticated official labels; no test path is accepted or discovered."""
    return load_labeled_json(
        official_training_path(repository_root),
        expected_sha256=OFFICIAL_SAMPLE_SHA256,
    )


def prepare_experiment(
    frame: Any,
    tokenizer: Any,
    *,
    serialization: str,
    field_aware: bool,
    config: V4TrainingConfig = V4TrainingConfig(),
) -> PreparedExperiment:
    """Apply the identical preprocessing path used by later train and inference calls."""
    encodings: list[EncodedV4Input] = []
    for row in frame.itertuples(index=False):
        prepared = prepare_v4_input(
            row.prompt_bn,
            row.context,
            row.response_bn,
            serialization=serialization,
        )
        encoded = (
            encode_field_aware(tokenizer, prepared, budget=config.field_budget)
            if field_aware
            else encode_comparison_baseline(
                tokenizer,
                prepared,
                maximum_length=config.maximum_length,
            )
        )
        encodings.append(encoded)
    folds = build_repeated_official_folds(frame)
    return PreparedExperiment(tuple(encodings), folds, truncation_statistics(encodings))


def context_presence(frame: Any) -> tuple[bool, ...]:
    """Return aggregate-safe context routing flags without serializing row text."""
    return tuple(raw_context_is_present(value) for value in frame["context"])
