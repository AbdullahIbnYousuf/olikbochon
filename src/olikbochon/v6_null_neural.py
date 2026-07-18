"""Frozen official-only neural candidates for the corrected context-absent route."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data_loading import validate_labeled_frame
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v3_modeling import resolve_faithful_logit_index
from .v3_training import (
    DynamicPairCollator,
    EncodedFrame,
    TokenizedRows,
    cleanup_cuda,
    set_all_seeds,
    subset_encoded,
)
from .v4_preprocessing import official_context_is_present
from .v4_training import load_offline_base
from .v4_validation import VALIDATION_SEEDS, select_threshold
from .v5_lexical import (
    build_v5_folds,
    deterministic_substring_prediction,
    fold_feasibility_audit,
)
from .v5_normalization import normalize_v5


EXPERIMENT_NAME = "v6_null_neural_baseline"
CANDIDATE_SET = "frozen"
CANDIDATE_J = "candidate_j_frozen_encoder_linear_probe"
CANDIDATE_K = "candidate_k_top_two_layers"
CANDIDATE_L = "candidate_l_full_finetune_control"
CANDIDATES = (CANDIDATE_J, CANDIDATE_K, CANDIDATE_L)
MAXIMUM_LENGTH = 192
MAXIMUM_EPOCHS = 6
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
EFFECTIVE_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
HEAD_LEARNING_RATE = 2e-5
TOP_LAYER_LEARNING_RATE = 5e-6
FULL_FINETUNE_LEARNING_RATE = 1e-5
MAXIMUM_GRADIENT_NORM = 1.0
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169
EXPECTED_ABSENT_LABELS = {0: 89, 1: 80}
NULL_CHAMPION = 0.574025
ROUTED_CHAMPION = 0.692051


@dataclass(frozen=True)
class CandidatePolicy:
    name: str
    encoder_policy: str
    head_learning_rate: float
    encoder_learning_rate: float | None


@dataclass(frozen=True)
class NullFoldResult:
    probabilities: np.ndarray
    validation_loss: float
    selected_epoch: int
    epoch_history: tuple[dict[str, Any], ...]
    optimizer_steps: int
    runtime_seconds: float
    peak_gpu_vram_bytes: int
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    trainable_parameters: int
    total_parameters: int
    reload: dict[str, Any]


POLICIES = {
    CANDIDATE_J: CandidatePolicy(CANDIDATE_J, "frozen_encoder_linear_head", 2e-5, None),
    CANDIDATE_K: CandidatePolicy(CANDIDATE_K, "top_two_encoder_layers_plus_head", 2e-5, 5e-6),
    CANDIDATE_L: CandidatePolicy(CANDIDATE_L, "full_finetune", 1e-5, 1e-5),
}


def serialize_null_input(prompt: Any, response: Any) -> str:
    """Serialize only normalized question and answer fields with ordinary markers."""
    return (
        f"[QUESTION]\n{normalize_v5(prompt)}\n\n"
        f"[ANSWER]\n{normalize_v5(response)}"
    )


def require_absent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a labeled fold and reject every context-present row."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    if any(official_context_is_present(value) for value in validated["context"]):
        raise ValueError("V6 neural fitting accepts corrected context-absent rows only")
    if set(validated["label"].unique()) != {0, 1}:
        raise ValueError("Every V6 neural train/validation fold must contain both labels")
    return validated


def encode_null_frame(
    frame: pd.DataFrame,
    tokenizer: Any,
    *,
    maximum_length: int = MAXIMUM_LENGTH,
) -> tuple[EncodedFrame, dict[str, int]]:
    """Tokenize the null-route serialization without retaining source strings."""
    if maximum_length != MAXIMUM_LENGTH:
        raise ValueError("V6 null-route maximum length is frozen at 192")
    validated = require_absent_frame(frame)
    features: list[dict[str, list[int]]] = []
    truncation_flags: list[bool] = []
    truncated = 0
    maximum_observed = 0
    for row in validated.itertuples(index=False):
        text = serialize_null_input(row.prompt_bn, row.response_bn)
        full = tokenizer(text, add_special_tokens=True, truncation=False)
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=maximum_length,
            padding=False,
        )
        item = {
            "input_ids": list(encoded["input_ids"]),
            "attention_mask": list(encoded["attention_mask"]),
        }
        if "token_type_ids" in encoded:
            item["token_type_ids"] = list(encoded["token_type_ids"])
        features.append(item)
        original_length = len(full["input_ids"])
        maximum_observed = max(maximum_observed, original_length)
        was_truncated = original_length > maximum_length
        truncation_flags.append(was_truncated)
        truncated += int(was_truncated)
    labels = tuple(int(value) for value in validated["label"])
    encoded_frame = EncodedFrame(
        tuple(features),
        labels,
        tuple(False for _ in features),
        tuple(truncation_flags),
        truncated,
    )
    return encoded_frame, {
        "rows": len(features),
        "maximum_length": maximum_length,
        "truncated_rows": truncated,
        "maximum_original_tokens": maximum_observed,
    }


def training_class_weights(labels: Sequence[int]) -> np.ndarray:
    """Compute balanced binary class weights from one training fold only."""
    truth = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(truth, minlength=2)
    if truth.size == 0 or set(np.unique(truth)) != {0, 1}:
        raise ValueError("Training-fold class weights require both labels")
    return (truth.size / (2.0 * counts)).astype(np.float32)


def _encoder_layers(model: Any) -> Any:
    base = getattr(model, "base_model", None)
    encoder = getattr(base, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None or len(layers) < 2:
        raise TypeError("Authenticated model does not expose at least two encoder layers")
    return layers


def _replace_with_linear_head(model: Any) -> None:
    import torch

    hidden = int(model.config.hidden_size)

    class LinearNullHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = torch.nn.Dropout(float(model.config.hidden_dropout_prob))
            self.out_proj = torch.nn.Linear(hidden, 2)

        def forward(self, features: Any) -> Any:
            return self.out_proj(self.dropout(features[:, 0, :]))

    model.classifier = LinearNullHead()


def configure_candidate(model: Any, candidate: str) -> dict[str, Any]:
    """Apply one exact trainability policy and return aggregate parameter counts."""
    if candidate not in POLICIES:
        raise ValueError(f"Unknown V6 candidate {candidate!r}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if candidate == CANDIDATE_J:
        _replace_with_linear_head(model)
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    elif candidate == CANDIDATE_K:
        layers = _encoder_layers(model)
        for layer in layers[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    trainable_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable_names:
        raise RuntimeError("V6 candidate policy produced no trainable parameters")
    return {
        "candidate": candidate,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        ),
        "trainable_tensor_count": len(trainable_names),
        "encoder_layer_count": len(_encoder_layers(model)),
        "trainable_parameter_names": trainable_names,
    }


def optimizer_groups(model: Any, candidate: str) -> list[dict[str, Any]]:
    """Construct the frozen candidate-specific AdamW learning-rate groups."""
    policy = POLICIES[candidate]
    head = [parameter for parameter in model.classifier.parameters() if parameter.requires_grad]
    if candidate == CANDIDATE_J:
        return [{"params": head, "lr": policy.head_learning_rate}]
    if candidate == CANDIDATE_K:
        head_ids = {id(parameter) for parameter in head}
        top = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        return [
            {"params": head, "lr": policy.head_learning_rate},
            {"params": top, "lr": policy.encoder_learning_rate},
        ]
    return [
        {
            "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
            "lr": policy.encoder_learning_rate,
        }
    ]


def _evaluate(
    model: Any,
    tokenizer: Any,
    encoded: EncodedFrame,
    class_weights: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    if encoded.labels is None:
        raise ValueError("V6 evaluation requires labels")
    loader = DataLoader(
        TokenizedRows(encoded),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=DynamicPairCollator(tokenizer),
    )
    device = torch.device("cuda")
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    faithful_index = resolve_faithful_logit_index(model.config, logits_dimension=2)
    probability_batches: list[np.ndarray] = []
    total_loss = 0.0
    examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**inputs)
                loss = loss_function(output.logits.float(), labels)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("V6 validation produced a non-finite loss")
            count = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * count
            examples += count
            probabilities = torch.softmax(output.logits.float(), dim=-1)[:, faithful_index]
            probability_batches.append(probabilities.detach().cpu().numpy())
    joined = np.concatenate(probability_batches).astype(np.float64)
    if not np.isfinite(joined).all():
        raise RuntimeError("V6 validation produced non-finite probabilities")
    predictions = predictions_from_label1(joined, 0.5)
    return joined, total_loss / examples, classification_metrics(encoded.labels, predictions)


def _save_delta_checkpoint(
    model: Any,
    checkpoint: Path,
    *,
    candidate: str,
    seed: int,
    selected_epoch: int,
) -> None:
    from safetensors.torch import save_file

    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    save_file(state, str(checkpoint / "model.safetensors"))
    metadata = (
        "{\n"
        f'  "candidate": "{candidate}",\n'
        f'  "seed": {seed},\n'
        f'  "selected_epoch": {selected_epoch},\n'
        '  "state_scope": "trainable_parameters_only"\n'
        "}\n"
    )
    (checkpoint / "checkpoint.json").write_text(metadata, encoding="utf-8")


def _load_delta_checkpoint(model_path: Path, checkpoint: Path, candidate: str) -> tuple[Any, Any]:
    from safetensors.torch import load_file

    tokenizer, model, _ = load_offline_base(model_path)
    audit = configure_candidate(model, candidate)
    state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    expected = set(audit["trainable_parameter_names"])
    if set(state) != expected:
        raise RuntimeError("V6 delta checkpoint does not match the candidate trainable state")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    }:
        raise RuntimeError("V6 delta checkpoint reload returned unexplained keys")
    return tokenizer, model


def _checkpoint_stats(checkpoint: Path) -> tuple[int, str]:
    weight = checkpoint / "model.safetensors"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    size = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    return size, digest


def train_null_fold(
    model_path: Path,
    train_data: EncodedFrame,
    validation_data: EncodedFrame,
    checkpoint: Path,
    *,
    candidate: str,
    seed: int,
) -> NullFoldResult:
    """Train one selected CUDA fold and verify its offline delta checkpoint exactly."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("V6 neural training requires CUDA; CPU fallback is prohibited")
    if checkpoint.exists():
        raise ValueError("V6 checkpoint directory must not already exist")
    if train_data.labels is None or validation_data.labels is None:
        raise ValueError("V6 training requires labeled train and validation data")
    if any(train_data.context_present) or any(validation_data.context_present):
        raise ValueError("A context-present row reached V6 neural fitting")
    if set(train_data.labels) != {0, 1} or set(validation_data.labels) != {0, 1}:
        raise ValueError("Every V6 neural fold must contain both labels")
    started = perf_counter()
    set_all_seeds(seed)
    tokenizer, model, _ = load_offline_base(model_path)
    parameter_audit = configure_candidate(model, candidate)
    class_weights = training_class_weights(train_data.labels)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TokenizedRows(train_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=DynamicPairCollator(tokenizer),
    )
    optimizer = torch.optim.AdamW(optimizer_groups(model, candidate), weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(loader) / GRADIENT_ACCUMULATION)
    total_updates = updates_per_epoch * MAXIMUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * WARMUP_FRACTION),
        num_training_steps=total_updates,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loss_function = torch.nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, Any]] = []
    best_rank: tuple[float, float, int] | None = None
    selected_epoch = 0
    selected_probabilities: np.ndarray | None = None
    selected_loss = math.inf
    optimizer_steps = 0
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        pending = 0
        for batch_index, batch in enumerate(loader):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                output = model(**inputs)
                loss = loss_function(output.logits.float(), labels)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("V6 training produced a non-finite loss")
            observed = float(loss.detach().cpu())
            epoch_loss += observed
            batch_count += 1
            pending += 1
            scaler.scale(loss / GRADIENT_ACCUMULATION).backward()
            if pending == GRADIENT_ACCUMULATION or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, MAXIMUM_GRADIENT_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                optimizer_steps += 1
        probabilities, validation_loss, metrics = _evaluate(
            model, tokenizer, validation_data, class_weights
        )
        rank = (float(metrics["macro_f1"]), -validation_loss, -epoch)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            selected_epoch = epoch
            selected_probabilities = probabilities.copy()
            selected_loss = validation_loss
            _save_delta_checkpoint(
                model,
                checkpoint,
                candidate=candidate,
                seed=seed,
                selected_epoch=epoch,
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / batch_count,
                "validation_loss": validation_loss,
                **metrics,
            }
        )
    if selected_probabilities is None or selected_epoch == 0:
        raise RuntimeError("V6 fold did not select a checkpoint")
    training_peak = int(torch.cuda.max_memory_allocated())
    del model, optimizer, scheduler, scaler, loader
    cleanup_cuda()
    set_all_seeds(seed)
    reload_tokenizer, reload_model = _load_delta_checkpoint(model_path, checkpoint, candidate)
    torch.cuda.reset_peak_memory_stats()
    reload_model.to(torch.device("cuda"))
    reloaded_probabilities, reloaded_loss, reloaded_metrics = _evaluate(
        reload_model, reload_tokenizer, validation_data, class_weights
    )
    reload_peak = int(torch.cuda.max_memory_allocated())
    expected_predictions = predictions_from_label1(selected_probabilities, 0.5)
    reloaded_predictions = predictions_from_label1(reloaded_probabilities, 0.5)
    probabilities_match = bool(np.array_equal(selected_probabilities, reloaded_probabilities))
    predictions_match = bool(np.array_equal(expected_predictions, reloaded_predictions))
    expected_metrics = classification_metrics(validation_data.labels, expected_predictions)
    metrics_match = expected_metrics == reloaded_metrics
    loss_match = bool(np.isclose(selected_loss, reloaded_loss, rtol=0, atol=1e-12))
    if not all((probabilities_match, predictions_match, metrics_match, loss_match)):
        raise RuntimeError("V6 checkpoint reload did not exactly reproduce the selected fold")
    checkpoint_size, checkpoint_digest = _checkpoint_stats(checkpoint)
    del reload_model, reload_tokenizer
    cleanup_cuda()
    return NullFoldResult(
        reloaded_probabilities,
        reloaded_loss,
        selected_epoch,
        tuple(history),
        optimizer_steps,
        perf_counter() - started,
        max(training_peak, reload_peak),
        checkpoint_size,
        checkpoint_digest,
        int(parameter_audit["trainable_parameters"]),
        int(parameter_audit["total_parameters"]),
        {
            "probabilities_match": True,
            "predictions_match": True,
            "metrics_match": True,
            "validation_loss_match": True,
        },
    )


def _prediction_counts(predictions: np.ndarray) -> dict[str, int]:
    counts = np.bincount(predictions, minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def _metric_record(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {**classification_metrics(truth, predictions), "prediction_counts": _prediction_counts(predictions)}


def _probability_summary(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    if truth.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("V6 probability summaries require finite aligned arrays")
    return {
        "brier_score": float(np.mean(np.square(probabilities - truth))),
        "probability_mean": float(probabilities.mean()),
        "probability_std": float(probabilities.std(ddof=0)),
        "probability_minimum": float(probabilities.min()),
        "probability_maximum": float(probabilities.max()),
    }


def _seed_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([record["macro_f1"] for record in records], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _encoded_truncation_audit(encoded: EncodedFrame) -> dict[str, int]:
    return {
        "rows": len(encoded.features),
        "maximum_length": MAXIMUM_LENGTH,
        "truncated_rows": int(encoded.response_fallback_count),
    }


def route_and_label_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Authenticate the frozen route totals using aggregate counts only."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    labels = validated["label"].to_numpy(dtype=np.int64)
    absent_counts = np.bincount(labels[~presence], minlength=2)
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V6 corrected route totals differ from the frozen 130/169 audit")
    if {0: int(absent_counts[0]), 1: int(absent_counts[1])} != EXPECTED_ABSENT_LABELS:
        raise RuntimeError("V6 context-absent label totals differ from the frozen 89/80 audit")
    return {
        "total_rows": len(validated),
        "context_present": int(presence.sum()),
        "context_absent": int((~presence).sum()),
        "context_absent_label_0": int(absent_counts[0]),
        "context_absent_label_1": int(absent_counts[1]),
    }


def run_v6_experiment(frame: pd.DataFrame, model_path: Path, output: Path) -> dict[str, Any]:
    """Execute only the three frozen V6 candidates and persist aggregate metadata."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    route_audit = route_and_label_audit(validated)
    folds = build_v5_folds(validated)
    feasibility = fold_feasibility_audit(validated, folds)
    if folds.seeds != VALIDATION_SEEDS or len(folds.folds) != 15:
        raise RuntimeError("V6 must use exactly 15 frozen repeated grouped folds")
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    truth = validated["label"].to_numpy(dtype=np.int64)
    groups = np.asarray(folds.audit.group_ids, dtype=object)
    absent_indices = np.flatnonzero(~presence)
    absent_positions = {
        int(original): position for position, original in enumerate(absent_indices)
    }
    set_all_seeds(VALIDATION_SEEDS[0])
    preparation_tokenizer, preparation_model, _ = load_offline_base(model_path)
    all_absent_encoded, input_preparation = encode_null_frame(
        validated.iloc[absent_indices].reset_index(drop=True), preparation_tokenizer
    )
    del preparation_model, preparation_tokenizer
    cleanup_cuda()
    output.mkdir(parents=True, exist_ok=False)
    present_predictions = np.asarray(
        [
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in validated.loc[presence].itertuples(index=False)
        ],
        dtype=np.int64,
    )
    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        oof = {
            seed: np.full(len(validated), np.nan, dtype=np.float64)
            for seed in VALIDATION_SEEDS
        }
        fold_records: list[dict[str, Any]] = []
        retained_bytes = 0
        transient_bytes = 0
        maximum_vram = 0
        for split in folds.folds:
            train = np.asarray(split.train_indices, dtype=np.int64)
            validation = np.asarray(split.validation_indices, dtype=np.int64)
            if set(groups[train]) & set(groups[validation]):
                raise RuntimeError("V6 group overlap detected")
            train_absent = train[~presence[train]]
            validation_absent = validation[~presence[validation]]
            train_encoded = subset_encoded(
                all_absent_encoded,
                np.asarray(
                    [absent_positions[int(index)] for index in train_absent],
                    dtype=np.int64,
                ),
            )
            validation_encoded = subset_encoded(
                all_absent_encoded,
                np.asarray(
                    [absent_positions[int(index)] for index in validation_absent],
                    dtype=np.int64,
                ),
            )
            checkpoint = (
                output
                / candidate
                / f"seed_{split.seed}"
                / f"fold_{split.fold}"
                / "checkpoint"
            )
            result = train_null_fold(
                model_path,
                train_encoded,
                validation_encoded,
                checkpoint,
                candidate=candidate,
                seed=split.seed,
            )
            oof[split.seed][validation_absent] = result.probabilities
            predictions_050 = predictions_from_label1(result.probabilities, 0.5)
            full_validation = validated.iloc[validation].reset_index(drop=True)
            full_presence = presence[validation]
            routed_fold_predictions = np.empty(validation.size, dtype=np.int64)
            routed_fold_predictions[~full_presence] = predictions_050
            routed_fold_predictions[full_presence] = [
                deterministic_substring_prediction(row.context, row.response_bn)
                for row in full_validation.loc[full_presence].itertuples(index=False)
            ]
            fold_record = {
                "candidate": candidate,
                "seed": split.seed,
                "fold": split.fold,
                "train_rows": int(train.size),
                "validation_rows": int(validation.size),
                "null_train_rows": int(train_absent.size),
                "null_validation_rows": int(validation_absent.size),
                "train_groups": int(len(set(groups[train]))),
                "validation_groups": int(len(set(groups[validation]))),
                "group_overlap_count": 0,
                "train_label_counts": np.bincount(
                    truth[train_absent], minlength=2
                ).astype(int).tolist(),
                "validation_label_counts": np.bincount(
                    truth[validation_absent], minlength=2
                ).astype(int).tolist(),
                "training_class_weights": training_class_weights(
                    truth[train_absent]
                ).astype(float).tolist(),
                "threshold_050": _metric_record(
                    truth[validation_absent], predictions_050
                ),
                "probability_summary": _probability_summary(
                    truth[validation_absent], result.probabilities
                ),
                "routed_threshold_050": {
                    **_metric_record(truth[validation], routed_fold_predictions),
                    "route_metrics": subgroup_metrics(
                        truth[validation], routed_fold_predictions, full_presence
                    ),
                },
                "selected_epoch": result.selected_epoch,
                "epoch_history": result.epoch_history,
                "validation_loss": result.validation_loss,
                "optimizer_steps": result.optimizer_steps,
                "runtime_seconds": result.runtime_seconds,
                "peak_gpu_vram_bytes": result.peak_gpu_vram_bytes,
                "checkpoint_size_bytes": result.checkpoint_size_bytes,
                "checkpoint_sha256": result.checkpoint_sha256,
                "checkpoint_retained": split.seed == 17 and split.fold == 1,
                "trainable_parameters": result.trainable_parameters,
                "total_parameters": result.total_parameters,
                "reload": result.reload,
                "train_truncation": _encoded_truncation_audit(train_encoded),
                "validation_truncation": _encoded_truncation_audit(validation_encoded),
            }
            fold_records.append(fold_record)
            print(
                f"V6 {candidate} seed={split.seed} fold={split.fold} "
                f"selected_epoch={result.selected_epoch} "
                f"null_macro_f1_050={fold_record['threshold_050']['macro_f1']:.6f} "
                f"runtime_seconds={result.runtime_seconds:.3f} "
                f"checkpoint_bytes={result.checkpoint_size_bytes}",
                flush=True,
            )
            metadata = checkpoint.parent / "fold_metrics.json"
            metadata.write_text(
                json.dumps(fold_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            transient_bytes += result.checkpoint_size_bytes
            maximum_vram = max(maximum_vram, result.peak_gpu_vram_bytes)
            if fold_record["checkpoint_retained"]:
                retained_bytes += result.checkpoint_size_bytes
            else:
                checkpoint.resolve().relative_to(output.resolve())
                shutil.rmtree(checkpoint)
        absent = ~presence
        if any(np.isnan(values[absent]).any() for values in oof.values()):
            raise RuntimeError("V6 null-route OOF coverage is incomplete")
        mean_null_probabilities = np.mean(
            np.stack([values[absent] for values in oof.values()]), axis=0
        )
        selected_threshold, selected_null_metrics = select_threshold(
            truth[absent], mean_null_probabilities
        )
        selected_null_predictions = predictions_from_label1(
            mean_null_probabilities, selected_threshold
        )
        routed_predictions = np.empty(len(validated), dtype=np.int64)
        routed_predictions[presence] = present_predictions
        routed_predictions[absent] = selected_null_predictions
        seed_null_records: list[dict[str, Any]] = []
        seed_routed_records: list[dict[str, Any]] = []
        for seed in VALIDATION_SEEDS:
            null_predictions = predictions_from_label1(oof[seed][absent], selected_threshold)
            seed_null_records.append(
                {"seed": seed, **_metric_record(truth[absent], null_predictions)}
            )
            seed_routed = np.empty(len(validated), dtype=np.int64)
            seed_routed[presence] = present_predictions
            seed_routed[absent] = null_predictions
            seed_routed_records.append(
                {"seed": seed, **_metric_record(truth, seed_routed)}
            )
        null_050_predictions = predictions_from_label1(mean_null_probabilities, 0.5)
        routed_050 = np.empty(len(validated), dtype=np.int64)
        routed_050[presence] = present_predictions
        routed_050[absent] = null_050_predictions
        selected_counts = np.bincount(selected_null_predictions, minlength=2)
        selected_share = float(selected_counts.max() / selected_counts.sum())
        selected_passes_guard = selected_share <= 0.90
        selected_null_macro = float(selected_null_metrics["macro_f1"])
        routed_selected_metrics = classification_metrics(truth, routed_predictions)
        candidates[candidate] = {
            "folds": fold_records,
            "null_route": {
                "threshold_050": _metric_record(truth[absent], null_050_predictions),
                "selected_threshold": selected_threshold,
                "selected": {
                    **selected_null_metrics,
                    "prediction_counts": _prediction_counts(selected_null_predictions),
                    "passes_class_collapse_guard": selected_passes_guard,
                },
                "probability_summary": _probability_summary(
                    truth[absent], mean_null_probabilities
                ),
                "seed_metrics": seed_null_records,
                "seed_macro_f1_distribution": _seed_distribution(seed_null_records),
                "gain_over_null_champion": float(
                    selected_null_metrics["macro_f1"] - NULL_CHAMPION
                ),
            },
            "routed": {
                "threshold_050": {
                    **_metric_record(truth, routed_050),
                    "route_metrics": subgroup_metrics(truth, routed_050, presence),
                },
                "selected": {
                    **routed_selected_metrics,
                    "prediction_counts": _prediction_counts(routed_predictions),
                    "route_metrics": subgroup_metrics(truth, routed_predictions, presence),
                },
                "seed_metrics": seed_routed_records,
                "seed_macro_f1_distribution": _seed_distribution(seed_routed_records),
                "gain_over_routed_champion": float(
                    routed_selected_metrics["macro_f1"] - ROUTED_CHAMPION
                ),
            },
            "acceptance": {
                "role": "diagnostic_only" if candidate == CANDIDATE_L else "primary",
                "null_pass_threshold": 0.58,
                "passes_null_gate": (
                    candidate != CANDIDATE_L
                    and selected_null_macro >= 0.58
                    and selected_passes_guard
                ),
                "beats_null_champion": selected_null_macro > NULL_CHAMPION,
                "beats_routed_champion": (
                    float(routed_selected_metrics["macro_f1"]) > ROUTED_CHAMPION
                ),
            },
            "maximum_peak_gpu_vram_bytes": maximum_vram,
            "transient_checkpoint_bytes": transient_bytes,
            "retained_checkpoint_count": 1,
            "retained_checkpoint_bytes": retained_bytes,
            "all_15_reloads_exact": all(
                all(record["reload"].values()) for record in fold_records
            ),
            "row_level_probabilities_persisted": False,
        }
    best_null = max(
        CANDIDATES,
        key=lambda name: candidates[name]["null_route"]["selected"]["macro_f1"],
    )
    best_routed = max(
        CANDIDATES,
        key=lambda name: candidates[name]["routed"]["selected"]["macro_f1"],
    )
    primary_passes = [
        name for name in (CANDIDATE_J, CANDIDATE_K) if candidates[name]["acceptance"]["passes_null_gate"]
    ]
    if primary_passes:
        next_direction = "sparse_plus_neural_hybrid"
    elif candidates[CANDIDATE_L]["null_route"]["selected"]["macro_f1"] >= 0.58:
        next_direction = "null_route_neural_model"
    else:
        next_direction = "retrieval_or_nli"
    return {
        "experiment": EXPERIMENT_NAME,
        "route_audit": route_audit,
        "input_preparation": input_preparation,
        "fold_feasibility": feasibility,
        "candidate_results": candidates,
        "candidate_count": len(CANDIDATES),
        "fits_per_candidate": 15,
        "total_fits": 45,
        "best_null_candidate": best_null,
        "best_routed_candidate": best_routed,
        "next_direction_by_predeclared_rule": next_direction,
        "raw_text_persisted": False,
        "row_level_values_persisted": False,
    }
