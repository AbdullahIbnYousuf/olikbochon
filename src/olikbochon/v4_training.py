"""Frozen Version 4 configuration and offline/artifact safety gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .v3_data import MODEL_REVISION, authenticate_model_directory
from .v3_training import load_base_classifier, load_saved_classifier, load_tokenizer
from .v4_preprocessing import FieldBudget
from .v4_validation import CANDIDATE_ORDER, FOLD_COUNT, THRESHOLD_GRID, VALIDATION_SEEDS


MODEL_SNAPSHOT = f"csebuetnlp/banglabert@{MODEL_REVISION}"
CHECKPOINT_POLICY = "validation_macro_f1_then_loss_then_earlier_epoch"


@dataclass(frozen=True)
class V4TrainingConfig:
    """Predeclared official-only experiment family."""

    candidates: tuple[str, ...] = CANDIDATE_ORDER
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS
    fold_count: int = FOLD_COUNT
    primary_metric: str = "macro_f1"
    threshold_grid: tuple[float, ...] = THRESHOLD_GRID
    maximum_length: int = 256
    field_budget: FieldBudget = field(default_factory=FieldBudget)
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation: int = 2
    learning_rate: float = 2e-5
    mixed_precision: str = "fp16"
    checkpoint_policy: str = CHECKPOINT_POLICY
    model_snapshot: str = MODEL_SNAPSHOT
    official_only: bool = True

    def __post_init__(self) -> None:
        if self.candidates != CANDIDATE_ORDER:
            raise ValueError("The initial V4 candidate family is frozen")
        if self.validation_seeds != VALIDATION_SEEDS or self.fold_count != FOLD_COUNT:
            raise ValueError("The V4 validation design is frozen")
        if self.threshold_grid != THRESHOLD_GRID:
            raise ValueError("The V4 threshold grid is frozen")
        if self.maximum_length != self.field_budget.maximum_length:
            raise ValueError("Training and field-aware maximum lengths must match")
        if not self.official_only:
            raise ValueError("Version 4 is official-only")


def require_local_model_path(path: Path) -> Path:
    """Reject remote identifiers and require a concrete local snapshot directory."""
    raw = str(path)
    if "://" in raw or raw.startswith(("hf:", "http:", "https:")):
        raise ValueError("Remote model identifiers are prohibited")
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_dir():
        raise ValueError("Model path must be an existing absolute local directory")
    return candidate.resolve()


def load_offline_base(model_path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Authenticate and load the base tokenizer/classifier without network access."""
    local = require_local_model_path(model_path)
    authentication = authenticate_model_directory(local)
    tokenizer = load_tokenizer(local)
    model, loading = load_base_classifier(local, tokenizer)
    return tokenizer, model, {"authentication": authentication, "loading": loading}


def load_offline_checkpoint(checkpoint: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load a saved V4 classifier strictly from a local directory."""
    local = require_local_model_path(checkpoint)
    tokenizer = load_tokenizer(local)
    model, loading = load_saved_classifier(local, tokenizer)
    return tokenizer, model, loading


def artifact_root(repository_root: Path) -> Path:
    """Return the only permitted V4 runtime artifact root after ignore validation."""
    root = Path(repository_root).resolve()
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        raise ValueError("Repository .gitignore is required")
    rules = {line.strip() for line in ignore_file.read_text(encoding="utf-8").splitlines()}
    if "artifacts/v4/" not in rules:
        raise ValueError("artifacts/v4/ must be explicitly ignored")
    return root / "artifacts" / "v4"


def resolve_artifact_path(repository_root: Path, relative_path: Path) -> Path:
    """Resolve a prospective output beneath the ignored V4 root without writing it."""
    root = artifact_root(repository_root)
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Artifact path escapes artifacts/v4") from exc
    return candidate
