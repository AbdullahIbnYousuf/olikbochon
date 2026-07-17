"""Manual PyTorch training utilities with authenticated loading and full OOM restarts."""

from __future__ import annotations

import gc
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v3_modeling import resolve_faithful_logit_index, validate_tokenizer_model_vocabulary
from .v3_preprocessing import build_transformer_pair, encode_pair_with_response_fallback


EXPECTED_BASE_MISSING = {
    "classifier.dense.bias",
    "classifier.dense.weight",
    "classifier.out_proj.bias",
    "classifier.out_proj.weight",
}
EXPECTED_BASE_UNEXPECTED = {
    "discriminator_predictions.dense.bias",
    "discriminator_predictions.dense.weight",
    "discriminator_predictions.dense_prediction.bias",
    "discriminator_predictions.dense_prediction.weight",
}
ID2LABEL = {0: "HALLUCINATED", 1: "FAITHFUL"}
LABEL2ID = {"HALLUCINATED": 0, "FAITHFUL": 1}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    learning_rate: float
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation: int = 2
    warmup_ratio: float = 0.10
    maximum_gradient_norm: float = 1.0
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    def oom_fallback(self) -> "TrainConfig":
        return TrainConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=4,
            gradient_accumulation=4,
            warmup_ratio=self.warmup_ratio,
            maximum_gradient_norm=self.maximum_gradient_norm,
            seed=self.seed,
        )


@dataclass(frozen=True)
class PhaseRestartRecord:
    phase: str
    original_batch_size: int
    original_accumulation: int
    oom_occurred: bool
    fallback_used: bool
    final_batch_size: int
    final_accumulation: int
    final_effective_batch_size: int
    complete_restart: bool
    failed_attempt_seconds: float
    successful_attempt_seconds: float


@dataclass(frozen=True)
class EncodedFrame:
    features: tuple[dict[str, list[int]], ...]
    labels: tuple[int, ...] | None
    context_present: tuple[bool, ...]
    response_fallback: tuple[bool, ...]
    response_fallback_count: int


@dataclass(frozen=True)
class EvaluationOutput:
    probabilities: np.ndarray
    validation_loss: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class TrainingOutput:
    evaluation: EvaluationOutput | None
    epoch_history: tuple[dict[str, Any], ...]
    selected_epoch: int | None
    update_steps: int
    maximum_allocated_bytes: int


@dataclass(frozen=True)
class CrossValidationOutput:
    probabilities: np.ndarray
    fold_metrics: tuple[dict[str, Any], ...]
    restart_records: tuple[PhaseRestartRecord, ...]
    maximum_allocated_bytes: int


def expected_optimizer_steps(batch_count: int, accumulation: int, epochs: int) -> int:
    """Count updates including the final incomplete accumulation window."""
    if batch_count <= 0 or accumulation <= 0 or epochs <= 0:
        raise ValueError("batch_count, accumulation, and epochs must be positive")
    return math.ceil(batch_count / accumulation) * epochs


def stage_a_rank(macro_f1: float, validation_loss: float, epoch: int) -> tuple[float, float, int]:
    """Rank Stage A checkpoints by macro F1, loss, then earlier epoch."""
    return float(macro_f1), -float(validation_loss), -int(epoch)


def place_oof_probabilities(
    destination: np.ndarray, validation_indices: np.ndarray, values: np.ndarray
) -> None:
    """Place one fold exactly once and reject overlap or shape errors."""
    indices = np.asarray(validation_indices, dtype=np.int64)
    probabilities = np.asarray(values, dtype=np.float64)
    if len(indices) != len(probabilities):
        raise ValueError("Validation indices and probabilities must have equal length")
    if np.isnan(destination[indices]).sum() != len(indices):
        raise ValueError("OOF destination positions were already populated")
    destination[indices] = probabilities


def subset_encoded(encoded: EncodedFrame, indices: np.ndarray) -> EncodedFrame:
    """Select encoded rows in the supplied deterministic order."""
    positions = [int(index) for index in indices]
    return EncodedFrame(
        tuple(encoded.features[index] for index in positions),
        (
            tuple(encoded.labels[index] for index in positions)
            if encoded.labels is not None
            else None
        ),
        tuple(encoded.context_present[index] for index in positions),
        tuple(encoded.response_fallback[index] for index in positions),
        sum(encoded.response_fallback[index] for index in positions),
    )


def set_all_seeds(seed: int) -> None:
    """Set every required RNG and deterministic CUDA controls."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_with_oom_restart(
    phase: str,
    config: TrainConfig,
    runner: Callable[[TrainConfig, int], Any],
    *,
    is_oom: Callable[[BaseException], bool],
    cleanup: Callable[[], None],
    seed_reset: Callable[[int], None],
) -> tuple[Any, PhaseRestartRecord]:
    """Run a phase once, then restart it completely once under the locked fallback."""
    started = perf_counter()
    try:
        output = runner(config, 1)
    except BaseException as error:
        failed_seconds = perf_counter() - started
        if not is_oom(error):
            raise
        cleanup()
        seed_reset(config.seed)
        fallback = config.oom_fallback()
        restarted = perf_counter()
        try:
            output = runner(fallback, 2)
        except BaseException as second_error:
            cleanup()
            if is_oom(second_error):
                raise RuntimeError(
                    f"{phase} exhausted its one full CUDA OOM restart"
                ) from second_error
            raise
        successful_seconds = perf_counter() - restarted
        return output, PhaseRestartRecord(
            phase,
            config.batch_size,
            config.gradient_accumulation,
            True,
            True,
            fallback.batch_size,
            fallback.gradient_accumulation,
            fallback.effective_batch_size,
            True,
            failed_seconds,
            successful_seconds,
        )
    successful_seconds = perf_counter() - started
    return output, PhaseRestartRecord(
        phase,
        config.batch_size,
        config.gradient_accumulation,
        False,
        False,
        config.batch_size,
        config.gradient_accumulation,
        config.effective_batch_size,
        False,
        0.0,
        successful_seconds,
    )


def encode_frame(frame: pd.DataFrame, tokenizer: Any, *, with_labels: bool) -> EncodedFrame:
    """Apply the locked pair preprocessing without padding or row-level logging."""
    features: list[dict[str, list[int]]] = []
    labels: list[int] = []
    contexts: list[bool] = []
    fallbacks: list[bool] = []
    fallback_count = 0
    for row in frame.itertuples(index=False):
        pair = build_transformer_pair(row.prompt_bn, row.context, row.response_bn)
        encoded = encode_pair_with_response_fallback(tokenizer, pair)
        item = {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
        }
        if encoded.token_type_ids is not None:
            item["token_type_ids"] = encoded.token_type_ids
        features.append(item)
        contexts.append(bool(pair.context_present))
        fallback_count += int(encoded.response_truncated)
        fallbacks.append(bool(encoded.response_truncated))
        if with_labels:
            labels.append(int(row.label))
    return EncodedFrame(
        tuple(features),
        tuple(labels) if with_labels else None,
        tuple(contexts),
        tuple(fallbacks),
        fallback_count,
    )


class TokenizedRows:
    def __init__(self, encoded: EncodedFrame) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = dict(self.encoded.features[index])
        if self.encoded.labels is not None:
            item["labels"] = self.encoded.labels[index]
        return item


class DynamicPairCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [int(row["labels"]) for row in rows] if "labels" in rows[0] else None
        features = [{key: value for key, value in row.items() if key != "labels"} for row in rows]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        if labels is not None:
            import torch

            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def _validate_loading_info(
    info: dict[str, Any], *, expected_missing: set[str], expected_unexpected: set[str]
) -> None:
    missing = set(info.get("missing_keys", []))
    unexpected = set(info.get("unexpected_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    errors = list(info.get("error_msgs", []))
    if any(key.startswith("electra.") for key in missing):
        raise RuntimeError("Missing reusable ELECTRA encoder parameters")
    if missing != expected_missing or unexpected != expected_unexpected:
        raise RuntimeError(
            f"Unexplained loading keys: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if mismatched or errors:
        raise RuntimeError(f"Model loading mismatch/errors: {mismatched!r} / {errors!r}")


def load_tokenizer(model_path: Path) -> Any:
    """Load only the authenticated local tokenizer with offline-safe arguments."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )


def load_base_classifier(model_path: Path, tokenizer: Any) -> tuple[Any, dict[str, Any]]:
    """Load and validate the expected pretraining-to-classification transition."""
    from transformers import AutoModelForSequenceClassification

    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    _validate_loading_info(
        info,
        expected_missing=EXPECTED_BASE_MISSING,
        expected_unexpected=EXPECTED_BASE_UNEXPECTED,
    )
    validate_tokenizer_model_vocabulary(tokenizer, model.config)
    resolve_faithful_logit_index(model.config, logits_dimension=2)
    return model, info


def load_saved_classifier(checkpoint: Path, tokenizer: Any) -> tuple[Any, dict[str, Any]]:
    """Reload a saved Stage A/final classifier with no unexplained keys."""
    from transformers import AutoModelForSequenceClassification

    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint),
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    _validate_loading_info(info, expected_missing=set(), expected_unexpected=set())
    if model.config.label2id != LABEL2ID or {
        int(key): value for key, value in model.config.id2label.items()
    } != ID2LABEL:
        raise RuntimeError("Saved classifier label mappings changed")
    validate_tokenizer_model_vocabulary(tokenizer, model.config)
    resolve_faithful_logit_index(model.config, logits_dimension=2)
    return model, info


def _evaluate(model: Any, loader: Any, *, device: Any) -> EvaluationOutput:
    import torch

    model.eval()
    probabilities: list[np.ndarray] = []
    losses = 0.0
    examples = 0
    faithful_index = resolve_faithful_logit_index(model.config, logits_dimension=2)
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.get("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**batch)
            batch_size = int(output.logits.shape[0])
            if labels is not None and output.loss is not None:
                losses += float(output.loss.detach().cpu()) * batch_size
                examples += batch_size
            probability = torch.softmax(output.logits.float(), dim=-1)[:, faithful_index]
            probabilities.append(probability.detach().cpu().numpy())
    joined = np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float64)
    encoded_labels = getattr(loader.dataset.encoded, "labels", None)
    truth = np.asarray(encoded_labels, dtype=np.int64) if encoded_labels is not None else None
    metrics = (
        classification_metrics(truth, predictions_from_label1(joined, 0.5))
        if truth is not None
        else {}
    )
    return EvaluationOutput(joined, losses / max(examples, 1), metrics)


def train_attempt(
    phase_name: str,
    start_checkpoint: Path,
    tokenizer: Any,
    train_data: EncodedFrame,
    validation_data: EncodedFrame | None,
    config: TrainConfig,
    *,
    base_checkpoint: bool,
    selected_checkpoint_directory: Path | None = None,
    final_checkpoint_directory: Path | None = None,
) -> TrainingOutput:
    """Train one complete fresh attempt and optionally retain the best epoch."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("Version 3 training requires CUDA; CPU fallback is prohibited")
    set_all_seeds(config.seed)
    model, _ = (
        load_base_classifier(start_checkpoint, tokenizer)
        if base_checkpoint
        else load_saved_classifier(start_checkpoint, tokenizer)
    )
    device = torch.device("cuda")
    model.to(device)
    generator = torch.Generator().manual_seed(config.seed)
    collator = DynamicPairCollator(tokenizer)
    train_loader = DataLoader(
        TokenizedRows(train_data),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
    )
    validation_loader = (
        DataLoader(
            TokenizedRows(validation_data),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        if validation_data is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_updates = expected_optimizer_steps(
        len(train_loader), config.gradient_accumulation, config.epochs
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, float, int] | None = None
    selected_epoch: int | None = None
    update_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                loss = model(**batch).loss / config.gradient_accumulation
            scaler.scale(loss).backward()
            should_step = (batch_index + 1) % config.gradient_accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.maximum_gradient_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_steps += 1
        evaluation = (
            _evaluate(model, validation_loader, device=device)
            if validation_loader is not None
            else None
        )
        row: dict[str, Any] = {"epoch": epoch}
        if evaluation is not None:
            row.update(
                {
                    "validation_loss": evaluation.validation_loss,
                    **evaluation.metrics,
                }
            )
            rank = stage_a_rank(
                evaluation.metrics["macro_f1"], evaluation.validation_loss, epoch
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                selected_epoch = epoch
                if selected_checkpoint_directory is not None:
                    if selected_checkpoint_directory.exists():
                        shutil.rmtree(selected_checkpoint_directory)
                    model.save_pretrained(selected_checkpoint_directory, safe_serialization=True)
                    tokenizer.save_pretrained(selected_checkpoint_directory)
        history.append(row)
        print(json.dumps({"phase": phase_name, **row}, sort_keys=True))
    final_evaluation = (
        _evaluate(model, validation_loader, device=device)
        if validation_loader is not None
        else None
    )
    if selected_checkpoint_directory is not None and selected_epoch is None:
        raise RuntimeError("Stage A did not select a checkpoint")
    if final_checkpoint_directory is not None:
        if final_checkpoint_directory.exists():
            shutil.rmtree(final_checkpoint_directory)
        model.save_pretrained(final_checkpoint_directory, safe_serialization=True)
        tokenizer.save_pretrained(final_checkpoint_directory)
    peak = int(torch.cuda.max_memory_allocated())
    del model, optimizer, scheduler, scaler, train_loader, validation_loader
    gc.collect()
    torch.cuda.empty_cache()
    return TrainingOutput(final_evaluation, tuple(history), selected_epoch, update_steps, peak)


def cleanup_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_probabilities(
    checkpoint: Path,
    tokenizer: Any,
    encoded: EncodedFrame,
    *,
    batch_size: int = 8,
) -> tuple[np.ndarray, int]:
    """Reload the frozen classifier and run aggregate-safe batched inference."""
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("Version 3 inference requires CUDA")
    model, _ = load_saved_classifier(checkpoint, tokenizer)
    model.to(torch.device("cuda"))
    loader = DataLoader(
        TokenizedRows(encoded),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DynamicPairCollator(tokenizer),
    )
    evaluation = _evaluate(model, loader, device=torch.device("cuda"))
    peak = int(torch.cuda.max_memory_allocated())
    del model, loader
    cleanup_cuda()
    return evaluation.probabilities, peak


def is_cuda_oom(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def run_cross_validation(
    arm_name: str,
    start_checkpoint: Path,
    tokenizer: Any,
    official_data: EncodedFrame,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    config: TrainConfig,
    *,
    base_checkpoint: bool,
) -> CrossValidationOutput:
    """Run independent fresh folds and place faithful probabilities in OOF order."""
    if official_data.labels is None:
        raise ValueError("Cross-validation requires labels")
    probabilities = np.full(len(official_data.features), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    records: list[PhaseRestartRecord] = []
    peak = 0
    for fold_index, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_split = subset_encoded(official_data, train_indices)
        validation_split = subset_encoded(official_data, validation_indices)

        def runner(active: TrainConfig, _attempt: int) -> TrainingOutput:
            return train_attempt(
                f"{arm_name}_fold_{fold_index}",
                start_checkpoint,
                tokenizer,
                train_split,
                validation_split,
                active,
                base_checkpoint=base_checkpoint,
            )

        output, record = run_with_oom_restart(
            f"{arm_name}_fold_{fold_index}",
            config,
            runner,
            is_oom=is_cuda_oom,
            cleanup=cleanup_cuda,
            seed_reset=set_all_seeds,
        )
        if output.evaluation is None:
            raise RuntimeError("Fold training did not return validation output")
        place_oof_probabilities(
            probabilities, validation_indices, output.evaluation.probabilities
        )
        predictions = predictions_from_label1(output.evaluation.probabilities, 0.5)
        counts = np.bincount(predictions, minlength=2)
        validation_truth = np.asarray(validation_split.labels, dtype=np.int64)
        validation_context = np.asarray(validation_split.context_present, dtype=bool)
        fold_rows.append(
            {
                "fold": fold_index,
                "validation_loss": output.evaluation.validation_loss,
                **output.evaluation.metrics,
                "prediction_counts": {0: int(counts[0]), 1: int(counts[1])},
                "context_present_count": int(sum(validation_split.context_present)),
                "context_absent_count": int(
                    len(validation_split.context_present) - sum(validation_split.context_present)
                ),
                "response_fallback_count": validation_split.response_fallback_count,
                "context_metrics": subgroup_metrics(
                    validation_truth, predictions, validation_context
                ),
            }
        )
        records.append(record)
        peak = max(peak, output.maximum_allocated_bytes)
    if np.isnan(probabilities).any():
        raise RuntimeError("OOF probabilities do not cover every official row exactly once")
    return CrossValidationOutput(probabilities, tuple(fold_rows), tuple(records), peak)
