"""Frozen Version 4 configuration and offline/artifact safety gates."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .metrics import classification_metrics, predictions_from_label1
from .v3_data import MODEL_REVISION, authenticate_model_directory
from .v3_modeling import resolve_faithful_logit_index
from .v3_training import (
    DynamicPairCollator,
    EncodedFrame,
    TokenizedRows,
    cleanup_cuda,
    load_base_classifier,
    load_saved_classifier,
    load_tokenizer,
    set_all_seeds,
)
from .v4_preprocessing import FieldBudget
from .v4_validation import CANDIDATE_ORDER, FOLD_COUNT, THRESHOLD_GRID, VALIDATION_SEEDS


MODEL_SNAPSHOT = f"csebuetnlp/banglabert@{MODEL_REVISION}"
CHECKPOINT_POLICY = "validation_macro_f1_then_loss_then_earlier_epoch"


@dataclass(frozen=True)
class SmokeEvaluation:
    """Aggregate evaluation output retained only for reload comparison."""

    probabilities: np.ndarray
    validation_loss: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SmokeTrainingResult:
    """Bounded single-run diagnostics without row text or identifiers."""

    evaluation: SmokeEvaluation
    training_loss: float
    initial_training_loss: float
    final_training_loss: float
    optimizer_steps: int
    training_rows_seen: int
    peak_gpu_vram_bytes: int


@dataclass(frozen=True)
class ReproductionFoldResult:
    """One selected grouped-fold checkpoint and aggregate-only diagnostics."""

    evaluation: SmokeEvaluation
    selected_epoch: int
    epoch_history: tuple[dict[str, Any], ...]
    initial_training_loss: float
    final_training_loss: float
    mean_training_loss: float
    optimizer_steps: int
    training_rows_seen: int
    runtime_seconds: float
    peak_gpu_vram_bytes: int
    checkpoint_size_bytes: int
    reload: dict[str, Any]


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


def encoded_frame(
    encodings: tuple[Any, ...], labels: tuple[int, ...], indices: tuple[int, ...]
) -> EncodedFrame:
    """Adapt prepared V4 encodings to the audited dynamic-padding training utility."""
    features: list[dict[str, list[int]]] = []
    contexts: list[bool] = []
    selected_labels: list[int] = []
    response_truncated: list[bool] = []
    for index in indices:
        row = encodings[index]
        item = {
            "input_ids": list(row.input_ids),
            "attention_mask": list(row.attention_mask),
        }
        if row.token_type_ids is not None:
            item["token_type_ids"] = list(row.token_type_ids)
        features.append(item)
        contexts.append(bool(row.context_present))
        selected_labels.append(int(labels[index]))
        response_truncated.append(
            row.lengths.response_retained < row.lengths.response_original
        )
    return EncodedFrame(
        tuple(features),
        tuple(selected_labels),
        tuple(contexts),
        tuple(response_truncated),
        sum(response_truncated),
    )


def _evaluate_smoke_model(
    model: Any,
    tokenizer: Any,
    encoded: EncodedFrame,
    *,
    batch_size: int,
) -> SmokeEvaluation:
    import torch
    from torch.utils.data import DataLoader

    if encoded.labels is None:
        raise ValueError("Smoke evaluation requires labels")
    device = torch.device("cuda")
    loader = DataLoader(
        TokenizedRows(encoded),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DynamicPairCollator(tokenizer),
    )
    faithful_index = resolve_faithful_logit_index(model.config, logits_dimension=2)
    probabilities: list[np.ndarray] = []
    total_loss = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**batch)
            if output.loss is None or not bool(torch.isfinite(output.loss).item()):
                raise RuntimeError("Smoke evaluation produced a non-finite loss")
            count = int(output.logits.shape[0])
            total_loss += float(output.loss.detach().cpu()) * count
            examples += count
            probability = torch.softmax(output.logits.float(), dim=-1)[:, faithful_index]
            probabilities.append(probability.detach().cpu().numpy())
    joined = np.concatenate(probabilities)
    predictions = predictions_from_label1(joined, 0.5)
    return SmokeEvaluation(
        joined,
        total_loss / examples,
        classification_metrics(encoded.labels, predictions),
    )


def train_smoke_model(
    model: Any,
    tokenizer: Any,
    train_data: EncodedFrame,
    validation_data: EncodedFrame,
    checkpoint_directory: Path,
    *,
    seed: int,
    max_steps: int,
    batch_size: int = 4,
    gradient_accumulation: int = 2,
    learning_rate: float = 2e-5,
) -> SmokeTrainingResult:
    """Run one capped CUDA/fp16 epoch and save one reloadable checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("V4 smoke training requires CUDA; CPU fallback is prohibited")
    if not 1 <= max_steps <= 20:
        raise ValueError("Smoke max_steps must be between 1 and 20")
    if checkpoint_directory.exists():
        raise ValueError("Smoke checkpoint directory must not already exist")
    set_all_seeds(seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TokenizedRows(train_data),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=DynamicPairCollator(tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=max_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss_total = 0.0
    batch_count = 0
    rows_seen = 0
    pending_batches = 0
    optimizer_steps = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            output = model(**batch)
            loss = output.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("Smoke training produced a non-finite loss")
        observed_loss = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = observed_loss
        final_loss = observed_loss
        loss_total += observed_loss
        batch_count += 1
        rows_seen += int(batch["labels"].shape[0])
        pending_batches += 1
        scaler.scale(loss / gradient_accumulation).backward()
        last_batch = batch_index + 1 == len(loader)
        if pending_batches == gradient_accumulation or last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending_batches = 0
            optimizer_steps += 1
            if optimizer_steps == max_steps:
                break
    if optimizer_steps == 0 or optimizer_steps > max_steps or initial_loss is None or final_loss is None:
        raise RuntimeError("Smoke training did not honor its optimizer-step cap")
    evaluation = _evaluate_smoke_model(
        model,
        tokenizer,
        validation_data,
        batch_size=batch_size,
    )
    checkpoint_directory.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_directory, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_directory)
    peak = int(torch.cuda.max_memory_allocated())
    del optimizer, scheduler, scaler, loader
    return SmokeTrainingResult(
        evaluation,
        loss_total / batch_count,
        initial_loss,
        final_loss,
        optimizer_steps,
        rows_seen,
        peak,
    )


def evaluate_offline_smoke_checkpoint(
    checkpoint: Path,
    validation_data: EncodedFrame,
    *,
    batch_size: int = 4,
) -> tuple[SmokeEvaluation, dict[str, Any], int]:
    """Reload through the V4 offline-safe path and evaluate the same labeled split."""
    import torch

    tokenizer, model, loading = load_offline_checkpoint(checkpoint)
    torch.cuda.reset_peak_memory_stats()
    model.to(torch.device("cuda"))
    evaluation = _evaluate_smoke_model(
        model,
        tokenizer,
        validation_data,
        batch_size=batch_size,
    )
    peak = int(torch.cuda.max_memory_allocated())
    del model, tokenizer
    cleanup_cuda()
    return evaluation, loading, peak


def train_reproduction_fold(
    model_path: Path,
    train_data: EncodedFrame,
    validation_data: EncodedFrame,
    checkpoint_directory: Path,
    *,
    seed: int,
    config: V4TrainingConfig = V4TrainingConfig(),
    checkpoint_selection: str = "best_validation",
) -> ReproductionFoldResult:
    """Train one frozen V4 fold from the authenticated base and verify its checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("V4 reproduction requires CUDA; CPU fallback is prohibited")
    if checkpoint_directory.exists():
        raise ValueError("Reproduction checkpoint directory must not already exist")
    if checkpoint_selection not in {"best_validation", "final_epoch"}:
        raise ValueError("Unknown fold checkpoint selection policy")
    started = perf_counter()
    tokenizer, model, _loading = load_offline_base(model_path)
    set_all_seeds(seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TokenizedRows(train_data),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=DynamicPairCollator(tokenizer),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    updates_per_epoch = math.ceil(len(loader) / config.gradient_accumulation)
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * 0.10),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    initial_loss: float | None = None
    final_loss: float | None = None
    all_loss_total = 0.0
    all_batch_count = 0
    optimizer_steps = 0
    rows_seen = 0
    best_rank: tuple[float, float, int] | None = None
    selected_epoch: int | None = None
    selected_evaluation: SmokeEvaluation | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending_batches = 0
        epoch_loss_total = 0.0
        epoch_batch_count = 0
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**batch)
                loss = output.loss
            if loss is None or not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Reproduction training produced a non-finite loss")
            observed_loss = float(loss.detach().cpu())
            if initial_loss is None:
                initial_loss = observed_loss
            final_loss = observed_loss
            epoch_loss_total += observed_loss
            epoch_batch_count += 1
            all_loss_total += observed_loss
            all_batch_count += 1
            rows_seen += int(batch["labels"].shape[0])
            pending_batches += 1
            scaler.scale(loss / config.gradient_accumulation).backward()
            last_batch = batch_index + 1 == len(loader)
            if pending_batches == config.gradient_accumulation or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending_batches = 0
                optimizer_steps += 1
        evaluation = _evaluate_smoke_model(
            model,
            tokenizer,
            validation_data,
            batch_size=config.batch_size,
        )
        rank = (
            float(evaluation.metrics["macro_f1"]),
            -float(evaluation.validation_loss),
            -epoch,
        )
        use_epoch = (
            epoch == config.epochs
            if checkpoint_selection == "final_epoch"
            else best_rank is None or rank > best_rank
        )
        if use_epoch:
            best_rank = rank
            selected_epoch = epoch
            selected_evaluation = evaluation
            if checkpoint_directory.exists():
                shutil.rmtree(checkpoint_directory)
            model.save_pretrained(checkpoint_directory, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_directory)
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss_total / epoch_batch_count,
                "validation_loss": evaluation.validation_loss,
                **evaluation.metrics,
            }
        )
    if initial_loss is None or final_loss is None or selected_epoch is None:
        raise RuntimeError("Reproduction fold did not complete its frozen training schedule")
    if selected_evaluation is None or optimizer_steps != total_updates:
        raise RuntimeError("Reproduction fold returned incomplete checkpoint state")
    training_peak = int(torch.cuda.max_memory_allocated())
    del model, optimizer, scheduler, scaler, loader
    cleanup_cuda()
    reloaded, reload_info, reload_peak = evaluate_offline_smoke_checkpoint(
        checkpoint_directory,
        validation_data,
        batch_size=config.batch_size,
    )
    selected_predictions = predictions_from_label1(selected_evaluation.probabilities, 0.5)
    reloaded_predictions = predictions_from_label1(reloaded.probabilities, 0.5)
    predictions_match = bool(np.array_equal(selected_predictions, reloaded_predictions))
    metrics_match = selected_evaluation.metrics == reloaded.metrics
    loss_match = bool(
        np.isclose(
            selected_evaluation.validation_loss,
            reloaded.validation_loss,
            rtol=0,
            atol=1e-12,
        )
    )
    if not predictions_match or not metrics_match or not loss_match:
        raise RuntimeError(
            "Reproduction checkpoint reload mismatch: "
            f"predictions={predictions_match}, metrics={metrics_match}, loss={loss_match}"
        )
    checkpoint_size = sum(
        path.stat().st_size for path in checkpoint_directory.rglob("*") if path.is_file()
    )
    return ReproductionFoldResult(
        reloaded,
        selected_epoch,
        tuple(history),
        initial_loss,
        final_loss,
        all_loss_total / all_batch_count,
        optimizer_steps,
        rows_seen,
        perf_counter() - started,
        max(training_peak, reload_peak),
        checkpoint_size,
        {
            "predictions_match": predictions_match,
            "metrics_match": metrics_match,
            "validation_loss_match": loss_match,
            "loading_missing_keys": len(reload_info.get("missing_keys", [])),
            "loading_unexpected_keys": len(reload_info.get("unexpected_keys", [])),
        },
    )


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
