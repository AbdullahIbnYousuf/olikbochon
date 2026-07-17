"""Official-only Version 4 preparation and narrowly bounded smoke CLI."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v3_training import encode_frame as encode_v3_frame
from .v3_training import subset_encoded
from .v4_diagnostics import probability_diagnostics, selected_epoch_distribution
from .v4_preprocessing import (
    V3_COMPATIBLE,
    EncodedV4Input,
    FieldBudget,
    encode_comparison_baseline,
    encode_field_aware,
    official_context_is_present,
    prepare_v4_input,
    truncation_statistics,
)
from .v4_training import (
    MODEL_SNAPSHOT,
    V4TrainingConfig,
    artifact_root,
    cleanup_cuda,
    encoded_frame,
    evaluate_offline_smoke_checkpoint,
    load_offline_base,
    load_offline_checkpoint,
    require_local_model_path,
    train_reproduction_fold,
    train_smoke_model,
)
from .v4_validation import (
    FOLD_COUNT,
    VALIDATION_SEEDS,
    RepeatedGroupedFolds,
    build_repeated_official_folds,
    fold_metric_record,
    route_threshold_diagnostics,
    select_threshold,
    summarize_fold_metrics,
)


SMOKE_CANDIDATE = "v3_compatible_baseline"
SMOKE_SEED = 17
SMOKE_MAXIMUM_LENGTH = 256
SMOKE_BATCH_SIZE = 4
SMOKE_GRADIENT_ACCUMULATION = 2
REPRODUCTION_OUTPUT_NAME = "v3_reproduction"
HISTORICAL_CANDIDATE = "v3_historical_control"
HISTORICAL_OUTPUT_NAME = "v3_historical_control"
HISTORICAL_MAXIMUM_LENGTH = 512
CORRECTED_CANDIDATE = "v4_schema_corrected_baseline"
CORRECTED_OUTPUT_NAME = "schema_corrected_baseline"


@dataclass(frozen=True)
class PreparedExperiment:
    """Aggregate-safe prepared inputs and frozen folds; no training state."""

    encodings: tuple[EncodedV4Input, ...]
    folds: RepeatedGroupedFolds
    truncation_summary: dict[str, Any]


def historical_control_config() -> V4TrainingConfig:
    """Return the exact predeclared historical schedule/length control."""
    return V4TrainingConfig(
        maximum_length=HISTORICAL_MAXIMUM_LENGTH,
        field_budget=FieldBudget(maximum_length=HISTORICAL_MAXIMUM_LENGTH),
        epochs=4,
        learning_rate=1e-5,
    )


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
    return tuple(official_context_is_present(value) for value in frame["context"])


def repository_root() -> Path:
    """Resolve the source checkout containing this module."""
    return Path(__file__).resolve().parents[2]


def require_approved_model_path(root: Path, model_path: Path) -> Path:
    """Require an authenticated local model beneath the ignored workspace model root."""
    local = require_local_model_path(model_path)
    approved_root = (Path(root).resolve() / "data" / "models").resolve()
    try:
        local.relative_to(approved_root)
    except ValueError as exc:
        raise ValueError("Model path must be inside the repository data/models directory") from exc
    return local


def require_smoke_output_path(root: Path, output_path: Path) -> Path:
    """Require a new smoke directory beneath the ignored V4 artifact root."""
    if not output_path.is_absolute():
        raise ValueError("Smoke output directory must be an absolute path")
    output = output_path.resolve()
    allowed = artifact_root(root).resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("Smoke output directory must be inside artifacts/v4") from exc
    if output.exists():
        raise ValueError("Smoke output directory must not already exist")
    return output


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the deliberately narrow smoke/reproduction interface."""
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v4_runner",
        description="Run a bounded official-only V4 CUDA workflow.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "smoke",
            "reproduce",
            "diagnose-reproduction",
            "historical-control",
            "corrected-baseline",
        ),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=(SMOKE_CANDIDATE, HISTORICAL_CANDIDATE, CORRECTED_CANDIDATE),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_smoke_arguments(args: argparse.Namespace, root: Path) -> tuple[Path, Path]:
    """Validate local-only inputs and hard smoke bounds before any data or CUDA work."""
    if args.mode != "smoke" or args.seed != SMOKE_SEED:
        raise ValueError("Smoke mode requires --seed 17")
    if args.candidate != SMOKE_CANDIDATE or args.max_length != SMOKE_MAXIMUM_LENGTH:
        raise ValueError("Smoke mode requires the V3-compatible candidate at length 256")
    if args.seeds is not None or args.folds is not None:
        raise ValueError("Smoke mode does not accept --seeds or --folds")
    if args.max_steps is None:
        raise ValueError("Smoke mode requires --max-steps")
    if not 1 <= args.max_steps <= 20:
        raise ValueError("--max-steps must be between 1 and 20")
    model_path = require_approved_model_path(root, args.model_path)
    output_path = require_smoke_output_path(root, args.output_dir)
    return model_path, output_path


def validate_reproduction_arguments(
    args: argparse.Namespace, root: Path
) -> tuple[Path, Path]:
    """Require the exact predeclared 15-fit V3-compatible reproduction."""
    if args.mode != "reproduce":
        raise ValueError("Reproduction validation requires --mode reproduce")
    if args.candidate != SMOKE_CANDIDATE or args.max_length != SMOKE_MAXIMUM_LENGTH:
        raise ValueError("Reproduce mode requires the V3-compatible candidate at length 256")
    if args.seed is not None or args.max_steps is not None:
        raise ValueError("Reproduce mode does not accept --seed or --max-steps")
    if tuple(args.seeds or ()) != VALIDATION_SEEDS:
        raise ValueError("Reproduce mode requires --seeds 17 29 43 in that order")
    if args.folds != FOLD_COUNT:
        raise ValueError("Reproduce mode requires --folds 5")
    model_path = require_approved_model_path(root, args.model_path)
    output_path = require_smoke_output_path(root, args.output_dir)
    expected_output = (artifact_root(root) / REPRODUCTION_OUTPUT_NAME).resolve()
    if output_path != expected_output:
        raise ValueError("Reproduce output directory must be artifacts/v4/v3_reproduction")
    return model_path, output_path


def validate_diagnostic_arguments(
    args: argparse.Namespace, root: Path
) -> tuple[Path, Path]:
    """Require the completed reproduction and its exact aggregate-only diagnostic scope."""
    if args.mode != "diagnose-reproduction":
        raise ValueError("Diagnostic validation requires --mode diagnose-reproduction")
    if args.candidate != SMOKE_CANDIDATE or args.max_length != SMOKE_MAXIMUM_LENGTH:
        raise ValueError("Diagnostics require the completed length-256 V3-compatible run")
    if args.seed is not None or args.max_steps is not None:
        raise ValueError("Diagnostic mode does not accept --seed or --max-steps")
    if tuple(args.seeds or ()) != VALIDATION_SEEDS or args.folds != FOLD_COUNT:
        raise ValueError("Diagnostics require --seeds 17 29 43 and --folds 5")
    model_path = require_approved_model_path(root, args.model_path)
    output_path = Path(args.output_dir).resolve()
    expected_output = (artifact_root(root) / REPRODUCTION_OUTPUT_NAME).resolve()
    if output_path != expected_output or not output_path.is_dir():
        raise ValueError("Diagnostics require the existing artifacts/v4/v3_reproduction")
    return model_path, output_path


def validate_historical_control_arguments(
    args: argparse.Namespace, root: Path
) -> tuple[Path, Path]:
    """Require one predeclared historical-schedule control with current grouped seeds."""
    if args.mode != "historical-control":
        raise ValueError("Historical control validation requires --mode historical-control")
    if args.candidate != HISTORICAL_CANDIDATE or args.max_length != HISTORICAL_MAXIMUM_LENGTH:
        raise ValueError("Historical control requires v3_historical_control at length 512")
    if args.seed is not None or args.max_steps is not None:
        raise ValueError("Historical control does not accept --seed or --max-steps")
    if tuple(args.seeds or ()) != VALIDATION_SEEDS or args.folds != FOLD_COUNT:
        raise ValueError("Historical control requires --seeds 17 29 43 and --folds 5")
    model_path = require_approved_model_path(root, args.model_path)
    output_path = require_smoke_output_path(root, args.output_dir)
    expected_output = (artifact_root(root) / HISTORICAL_OUTPUT_NAME).resolve()
    if output_path != expected_output:
        raise ValueError("Historical control output must be artifacts/v4/v3_historical_control")
    return model_path, output_path


def validate_corrected_baseline_arguments(
    args: argparse.Namespace, root: Path
) -> tuple[Path, Path]:
    """Require the exact schema-corrected 15-fit comparison and new artifact family."""
    if args.mode != "corrected-baseline":
        raise ValueError("Corrected baseline validation requires --mode corrected-baseline")
    if args.candidate != CORRECTED_CANDIDATE or args.max_length != SMOKE_MAXIMUM_LENGTH:
        raise ValueError(
            "Corrected baseline requires v4_schema_corrected_baseline at length 256"
        )
    if args.seed is not None or args.max_steps is not None:
        raise ValueError("Corrected baseline does not accept --seed or --max-steps")
    if tuple(args.seeds or ()) != VALIDATION_SEEDS:
        raise ValueError("Corrected baseline requires --seeds 17 29 43 in that order")
    if args.folds != FOLD_COUNT:
        raise ValueError("Corrected baseline requires --folds 5")
    model_path = require_approved_model_path(root, args.model_path)
    output_path = require_smoke_output_path(root, args.output_dir)
    expected_output = (artifact_root(root) / CORRECTED_OUTPUT_NAME).resolve()
    if output_path != expected_output:
        raise ValueError(
            "Corrected baseline output must be artifacts/v4/schema_corrected_baseline"
        )
    return model_path, output_path


def route_coverage_audit(
    labels: Any,
    context_flags: Any,
    folds: RepeatedGroupedFolds,
) -> dict[str, Any]:
    """Validate aggregate schema routes and complete per-seed validation coverage."""
    truth = np.asarray(labels, dtype=np.int64)
    presence = np.asarray(context_flags, dtype=bool)
    if truth.size == 0 or truth.size != presence.size:
        raise RuntimeError("Corrected route labels and flags must be nonempty and aligned")
    if set(np.unique(truth)) != {0, 1}:
        raise RuntimeError("Corrected route audit requires both official labels")

    def counts(mask: np.ndarray) -> dict[str, int]:
        route_labels = np.bincount(truth[mask], minlength=2)
        return {
            "rows": int(mask.sum()),
            "label_0": int(route_labels[0]),
            "label_1": int(route_labels[1]),
        }

    route_counts = {
        "context_present": counts(presence),
        "context_absent": counts(~presence),
    }
    seed_coverage: dict[str, dict[str, Any]] = {}
    for seed in folds.seeds:
        assignments = np.zeros(truth.size, dtype=np.int64)
        for fold in folds.folds:
            if fold.seed == seed:
                assignments[list(fold.validation_indices)] += 1
        if not np.all(assignments == 1):
            raise RuntimeError(f"Corrected route coverage is incomplete for seed {seed}")
        seed_coverage[str(seed)] = {
            "complete_oof_coverage": True,
            **route_counts,
        }
    return {
        "total_rows": int(truth.size),
        **route_counts,
        "per_seed_validation_coverage": seed_coverage,
        "row_level_values_persisted": False,
    }


def retain_selected_checkpoint(mode: str, seed: int, fold: int) -> bool:
    """Apply the compact retention policy only to its predeclared experiment modes."""
    if mode not in {"historical-control", "corrected-baseline"}:
        return True
    return seed == VALIDATION_SEEDS[0] and fold == 1


def split_distribution(labels: Any, context_flags: Any, indices: Any) -> dict[str, Any]:
    """Return aggregate label and route counts for one train or validation split."""
    truth = np.asarray(labels, dtype=np.int64)[indices]
    presence = np.asarray(context_flags, dtype=bool)[indices]

    def route(mask: np.ndarray) -> dict[str, int]:
        counts = np.bincount(truth[mask], minlength=2)
        return {
            "rows": int(mask.sum()),
            "label_0": int(counts[0]),
            "label_1": int(counts[1]),
        }

    label_counts = np.bincount(truth, minlength=2)
    return {
        "rows": int(truth.size),
        "label_0": int(label_counts[0]),
        "label_1": int(label_counts[1]),
        "context_present": route(presence),
        "context_absent": route(~presence),
    }


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_smoke(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Execute exactly one V3-compatible seed-17 fold smoke run."""
    import torch

    model_path, output_path = validate_smoke_arguments(args, root)
    if not torch.cuda.is_available():
        raise RuntimeError("V4 smoke training requires CUDA; CPU fallback is prohibited")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    started = perf_counter()
    tokenizer, model, model_loading = load_offline_base(model_path)
    frame = load_official_training_frame(root)
    config = V4TrainingConfig(maximum_length=args.max_length)
    prepared = prepare_experiment(
        frame,
        tokenizer,
        serialization=V3_COMPATIBLE,
        field_aware=False,
        config=config,
    )
    fold = next(
        item for item in prepared.folds.folds if item.seed == args.seed and item.fold == 1
    )
    group_ids = np.asarray(prepared.folds.audit.group_ids, dtype=object)
    if set(group_ids[list(fold.train_indices)]) & set(
        group_ids[list(fold.validation_indices)]
    ):
        raise RuntimeError("Group leakage detected in the selected smoke split")
    labels = tuple(int(value) for value in frame["label"])
    train_data = encoded_frame(prepared.encodings, labels, fold.train_indices)
    validation_data = encoded_frame(prepared.encodings, labels, fold.validation_indices)
    checkpoint = output_path / "checkpoint"
    output_path.mkdir(parents=True)
    result = train_smoke_model(
        model,
        tokenizer,
        train_data,
        validation_data,
        checkpoint,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=SMOKE_BATCH_SIZE,
        gradient_accumulation=SMOKE_GRADIENT_ACCUMULATION,
    )
    del model, tokenizer
    gc.collect()
    cleanup_cuda()
    reloaded, reload_info, reload_peak = evaluate_offline_smoke_checkpoint(
        checkpoint,
        validation_data,
        batch_size=SMOKE_BATCH_SIZE,
    )
    before_predictions = predictions_from_label1(result.evaluation.probabilities, 0.5)
    after_predictions = predictions_from_label1(reloaded.probabilities, 0.5)
    predictions_match = bool(np.array_equal(before_predictions, after_predictions))
    metrics_match = result.evaluation.metrics == reloaded.metrics
    loss_match = bool(
        np.isclose(result.evaluation.validation_loss, reloaded.validation_loss, rtol=0, atol=1e-12)
    )
    if not predictions_match or not metrics_match or not loss_match:
        raise RuntimeError(
            "Reload mismatch: "
            f"predictions={predictions_match}, metrics={metrics_match}, loss={loss_match}"
        )
    truth = np.asarray(validation_data.labels, dtype=np.int64)
    contexts = np.asarray(validation_data.context_present, dtype=bool)
    metrics = {
        "threshold": 0.5,
        "validation_loss": result.evaluation.validation_loss,
        **result.evaluation.metrics,
        "context_metrics": subgroup_metrics(truth, before_predictions, contexts),
    }
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    runtime = {
        "wall_clock_seconds": perf_counter() - started,
        "peak_gpu_vram_bytes": max(result.peak_gpu_vram_bytes, reload_peak),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    smoke_config = {
        "mode": "smoke",
        "candidate": args.candidate,
        "model_snapshot": MODEL_SNAPSHOT,
        "model_path_role": "repository_ignored_authenticated_snapshot",
        "seed": args.seed,
        "fold": fold.fold,
        "maximum_length": args.max_length,
        "max_optimizer_steps": args.max_steps,
        "actual_optimizer_steps": result.optimizer_steps,
        "batch_size": SMOKE_BATCH_SIZE,
        "gradient_accumulation": SMOKE_GRADIENT_ACCUMULATION,
        "precision": "cuda_fp16",
        "threshold": 0.5,
        "split_train_rows": len(fold.train_indices),
        "training_rows_seen": result.training_rows_seen,
        "validation_rows": len(fold.validation_indices),
        "total_group_count": prepared.folds.audit.group_count,
        "validation_group_count": len(set(group_ids[list(fold.validation_indices)])),
        "git_commit": _git_commit(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    reload_record = {
        "predictions_match": predictions_match,
        "metrics_match": metrics_match,
        "validation_loss_match": loss_match,
        "loading_missing_keys": len(reload_info.get("missing_keys", [])),
        "loading_unexpected_keys": len(reload_info.get("unexpected_keys", [])),
    }
    _write_json(output_path / "config.json", smoke_config)
    _write_json(
        output_path / "metrics.json",
        {
            "training_loss": result.training_loss,
            "initial_training_loss": result.initial_training_loss,
            "final_training_loss": result.final_training_loss,
            **metrics,
            "checkpoint_size_bytes": checkpoint_bytes,
            "reload": reload_record,
        },
    )
    _write_json(output_path / "truncation.json", prepared.truncation_summary)
    _write_json(output_path / "runtime.json", runtime)
    _write_json(
        output_path / "log.json",
        {
            "status": "passed",
            "official_rows_loaded": len(frame),
            "model_revision": model_loading["authentication"]["revision"],
            "weight_sha256": model_loading["authentication"]["weight_sha256"],
        },
    )
    return {
        "status": "passed",
        "config": smoke_config,
        "metrics": metrics,
        "runtime": runtime,
        "reload": reload_record,
        "checkpoint_size_bytes": checkpoint_bytes,
    }


def run_reproduction_diagnostics(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Recompute only aggregate calibration facts from the 15 selected checkpoints."""
    import torch

    _model_path, output_path = validate_diagnostic_arguments(args, root)
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint diagnostics require CUDA; CPU fallback is prohibited")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    saved_records = _read_json(output_path / "fold_metrics.json")["folds"]
    if len(saved_records) != len(VALIDATION_SEEDS) * FOLD_COUNT:
        raise RuntimeError("Completed reproduction does not contain exactly 15 fold records")
    by_fold = {(int(row["seed"]), int(row["fold"])): row for row in saved_records}
    first_checkpoint = output_path / "seed_17" / "fold_1" / "checkpoint"
    tokenizer, preparation_model, _loading = load_offline_checkpoint(first_checkpoint)
    frame = load_official_training_frame(root)
    config = V4TrainingConfig(maximum_length=SMOKE_MAXIMUM_LENGTH)
    prepared = prepare_experiment(
        frame,
        tokenizer,
        serialization=V3_COMPATIBLE,
        field_aware=False,
        config=config,
    )
    del preparation_model, tokenizer
    gc.collect()
    cleanup_cuda()
    labels = tuple(int(value) for value in frame["label"])
    truth = np.asarray(labels, dtype=np.int64)
    diagnostics: list[dict[str, Any]] = []
    for fold in prepared.folds.folds:
        saved = by_fold[(fold.seed, fold.fold)]
        validation_data = encoded_frame(
            prepared.encodings,
            labels,
            fold.validation_indices,
        )
        checkpoint = (
            output_path / f"seed_{fold.seed}" / f"fold_{fold.fold}" / "checkpoint"
        )
        evaluation, loading, _peak = evaluate_offline_smoke_checkpoint(
            checkpoint,
            validation_data,
            batch_size=config.batch_size,
        )
        saved_metrics = {
            key: saved[key]
            for key in ("macro_f1", "f1_label0", "f1_label1", "accuracy", "confusion_matrix")
        }
        loss_match = bool(
            np.isclose(
                evaluation.validation_loss,
                float(saved["validation_loss"]),
                rtol=0,
                atol=1e-12,
            )
        )
        if evaluation.metrics != saved_metrics or not loss_match:
            raise RuntimeError(
                f"Diagnostic checkpoint mismatch for seed {fold.seed}, fold {fold.fold}"
            )
        validation_indices = np.asarray(fold.validation_indices, dtype=np.int64)
        row = probability_diagnostics(
            truth[validation_indices],
            evaluation.probabilities,
        )
        row.update(
            {
                "seed": fold.seed,
                "fold": fold.fold,
                "selected_epoch": int(saved["selected_epoch"]),
                "validation_loss": evaluation.validation_loss,
                "epoch_history": saved["epoch_history"],
                "checkpoint_metrics_match": True,
                "checkpoint_validation_loss_match": True,
                "loading_missing_keys": len(loading.get("missing_keys", [])),
                "loading_unexpected_keys": len(loading.get("unexpected_keys", [])),
            }
        )
        diagnostics.append(row)
    best_thresholds = [float(row["best_frozen_grid"]["threshold"]) for row in diagnostics]
    brier_scores = np.asarray([float(row["brier_score"]) for row in diagnostics])
    near_optimal_count = sum(bool(row["threshold_054_near_optimal"]) for row in diagnostics)
    report = {
        "scope": "aggregate-only offline recomputation from selected fold checkpoints",
        "fold_count": len(diagnostics),
        "selected_epoch_distribution": selected_epoch_distribution(saved_records),
        "best_threshold_distribution": {
            f"{threshold:.2f}": best_thresholds.count(threshold)
            for threshold in sorted(set(best_thresholds))
        },
        "threshold_054_near_optimal_definition": (
            "best-grid threshold within 0.02 and macro-F1 gap from best no greater than 0.02"
        ),
        "threshold_054_near_optimal_count": near_optimal_count,
        "threshold_054_near_optimal_fraction": near_optimal_count / len(diagnostics),
        "brier_score_summary": {
            "mean": float(brier_scores.mean()),
            "std": float(brier_scores.std(ddof=0)),
            "minimum": float(brier_scores.min()),
            "maximum": float(brier_scores.max()),
        },
        "folds": diagnostics,
        "row_level_probabilities_persisted": False,
        "training_performed": False,
    }
    _write_json(output_path / "calibration_diagnostics.json", report)
    return {
        "status": "passed",
        "fold_count": len(diagnostics),
        "selected_epoch_distribution": report["selected_epoch_distribution"],
        "best_threshold_distribution": report["best_threshold_distribution"],
        "threshold_054_near_optimal_count": near_optimal_count,
        "brier_score_summary": report["brier_score_summary"],
    }


def run_reproduction(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Run one exact predeclared repeated grouped control family."""
    import torch

    historical = args.mode == "historical-control"
    corrected = args.mode == "corrected-baseline"
    if historical:
        model_path, output_path = validate_historical_control_arguments(args, root)
    elif corrected:
        model_path, output_path = validate_corrected_baseline_arguments(args, root)
    else:
        model_path, output_path = validate_reproduction_arguments(args, root)
    if not torch.cuda.is_available():
        raise RuntimeError("V4 reproduction requires CUDA; CPU fallback is prohibited")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    started = perf_counter()
    tokenizer, preparation_model, model_loading = load_offline_base(model_path)
    frame = load_official_training_frame(root)
    labels = tuple(int(value) for value in frame["label"])
    if historical:
        config = historical_control_config()
        folds = build_repeated_official_folds(frame)
        all_encoded = encode_v3_frame(frame, tokenizer, with_labels=True)
        truncation_summary = {
            "rows": len(frame),
            "historical_v3_response_fallback_count": all_encoded.response_fallback_count,
            "maximum_length": HISTORICAL_MAXIMUM_LENGTH,
        }
        candidate_name = HISTORICAL_CANDIDATE
    else:
        config = V4TrainingConfig(maximum_length=args.max_length)
        prepared = prepare_experiment(
            frame,
            tokenizer,
            serialization=V3_COMPATIBLE,
            field_aware=False,
            config=config,
        )
        folds = prepared.folds
        all_encoded = encoded_frame(
            prepared.encodings,
            labels,
            tuple(range(len(frame))),
        )
        truncation_summary = prepared.truncation_summary
        candidate_name = CORRECTED_CANDIDATE if corrected else SMOKE_CANDIDATE
    del preparation_model, tokenizer
    gc.collect()
    cleanup_cuda()
    if folds.seeds != VALIDATION_SEEDS or len(folds.folds) != 15:
        raise RuntimeError("Frozen repeated grouped folds were not reproduced exactly")
    truth = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(folds.audit.group_ids, dtype=object)
    contexts = np.asarray(all_encoded.context_present, dtype=bool)
    route_audit = None
    if corrected:
        independently_detected = np.asarray(context_presence(frame), dtype=bool)
        if not np.array_equal(contexts, independently_detected):
            raise RuntimeError("Corrected route flags differ between preparation and routing")
        route_audit = route_coverage_audit(truth, contexts, folds)
    output_path.mkdir(parents=True)
    fold_metadata_root = output_path / "fold_metadata"
    fold_metadata_root.mkdir()
    oof_by_seed = {
        seed: np.full(len(frame), np.nan, dtype=np.float64) for seed in VALIDATION_SEEDS
    }
    fold_records: list[dict[str, Any]] = []
    checkpoint_total_bytes = 0
    retained_checkpoint_bytes = 0
    maximum_peak = 0
    for fold in folds.folds:
        train_groups = set(group_ids[list(fold.train_indices)])
        validation_groups = set(group_ids[list(fold.validation_indices)])
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(
                f"Group leakage detected for seed {fold.seed}, fold {fold.fold}"
            )
        train_data = subset_encoded(all_encoded, np.asarray(fold.train_indices, dtype=np.int64))
        validation_data = subset_encoded(
            all_encoded,
            np.asarray(fold.validation_indices, dtype=np.int64),
        )
        checkpoint = (
            output_path / f"seed_{fold.seed}" / f"fold_{fold.fold}" / "checkpoint"
        )
        result = train_reproduction_fold(
            model_path,
            train_data,
            validation_data,
            checkpoint,
            seed=fold.seed,
            config=config,
            checkpoint_selection="final_epoch" if historical else "best_validation",
        )
        target = oof_by_seed[fold.seed]
        validation_indices = np.asarray(fold.validation_indices, dtype=np.int64)
        if not np.isnan(target[validation_indices]).all():
            raise RuntimeError("A reproduction OOF position was assigned more than once")
        target[validation_indices] = result.evaluation.probabilities
        predictions = predictions_from_label1(result.evaluation.probabilities, 0.5)
        metric_record = fold_metric_record(
            truth[validation_indices],
            predictions,
            contexts[validation_indices],
            seed=fold.seed,
            fold=fold.fold,
        )
        prediction_counts = np.bincount(predictions, minlength=2)
        fold_record = {
            **metric_record,
            "prediction_counts": {
                "0": int(prediction_counts[0]),
                "1": int(prediction_counts[1]),
            },
            "validation_loss": result.evaluation.validation_loss,
            "train_rows": len(fold.train_indices),
            "validation_rows": len(fold.validation_indices),
            "train_distribution": split_distribution(
                truth,
                contexts,
                np.asarray(fold.train_indices, dtype=np.int64),
            ),
            "validation_distribution": split_distribution(
                truth,
                contexts,
                validation_indices,
            ),
            "train_groups": len(train_groups),
            "validation_groups": len(validation_groups),
            "group_overlap_count": 0,
            "selected_epoch": result.selected_epoch,
            "initial_training_loss": result.initial_training_loss,
            "final_training_loss": result.final_training_loss,
            "mean_training_loss": result.mean_training_loss,
            "optimizer_steps": result.optimizer_steps,
            "training_rows_seen": result.training_rows_seen,
            "runtime_seconds": result.runtime_seconds,
            "peak_gpu_vram_bytes": result.peak_gpu_vram_bytes,
            "checkpoint_size_bytes": result.checkpoint_size_bytes,
            "checkpoint_weight_sha256": sha256_file(checkpoint / "model.safetensors"),
            "checkpoint_role": f"seed_{fold.seed}/fold_{fold.fold}/checkpoint",
            "checkpoint_retained": retain_selected_checkpoint(
                args.mode,
                fold.seed,
                fold.fold,
            ),
            "epoch_history": result.epoch_history,
            "reload": result.reload,
        }
        fold_records.append(fold_record)
        checkpoint_total_bytes += result.checkpoint_size_bytes
        maximum_peak = max(maximum_peak, result.peak_gpu_vram_bytes)
        _write_json(
            fold_metadata_root / f"seed_{fold.seed}_fold_{fold.fold}.json",
            fold_record,
        )
        if fold_record["checkpoint_retained"]:
            retained_checkpoint_bytes += result.checkpoint_size_bytes
        else:
            checkpoint.resolve().relative_to(output_path.resolve())
            shutil.rmtree(checkpoint)
    if any(np.isnan(values).any() for values in oof_by_seed.values()):
        raise RuntimeError("Reproduction OOF probabilities do not cover every official row")
    mean_probabilities = np.mean(np.stack(tuple(oof_by_seed.values())), axis=0)
    predictions_050 = predictions_from_label1(mean_probabilities, 0.5)
    metrics_050 = classification_metrics(truth, predictions_050)
    counts_050 = np.bincount(predictions_050, minlength=2)
    candidate_share = float(counts_050.max() / len(predictions_050))
    selected_threshold, selected_metrics = select_threshold(truth, mean_probabilities)
    selected_predictions = predictions_from_label1(mean_probabilities, selected_threshold)
    selected_counts = np.bincount(selected_predictions, minlength=2)
    seed_metrics: list[dict[str, Any]] = []
    seed_scores: list[float] = []
    for seed in VALIDATION_SEEDS:
        probabilities = oof_by_seed[seed]
        predictions_at_050 = predictions_from_label1(probabilities, 0.5)
        predictions_at_selected = predictions_from_label1(
            probabilities,
            selected_threshold,
        )
        at_050 = classification_metrics(truth, predictions_at_050)
        at_selected = classification_metrics(
            truth,
            predictions_at_selected,
        )
        counts_at_050 = np.bincount(predictions_at_050, minlength=2)
        counts_at_selected = np.bincount(predictions_at_selected, minlength=2)
        seed_scores.append(float(at_selected["macro_f1"]))
        seed_metrics.append(
            {
                "seed": seed,
                "oof_rows": len(probabilities),
                "oof_complete": True,
                "threshold_050": {
                    **at_050,
                    "prediction_counts": {
                        "0": int(counts_at_050[0]),
                        "1": int(counts_at_050[1]),
                    },
                    "context_metrics": subgroup_metrics(
                        truth,
                        predictions_at_050,
                        contexts,
                    ),
                },
                "selected_threshold": selected_threshold,
                "selected_threshold_metrics": {
                    **at_selected,
                    "prediction_counts": {
                        "0": int(counts_at_selected[0]),
                        "1": int(counts_at_selected[1]),
                    },
                    "context_metrics": subgroup_metrics(
                        truth,
                        predictions_at_selected,
                        contexts,
                    ),
                },
            }
        )
    threshold_report = {
        "procedure": "mean the three complete official-only OOF probabilities per row, then apply the frozen grid",
        "threshold_050": {
            **metrics_050,
            "prediction_counts": {"0": int(counts_050[0]), "1": int(counts_050[1])},
            "context_metrics": subgroup_metrics(truth, predictions_050, contexts),
        },
        "selected_threshold": selected_threshold,
        "selected_threshold_metrics": {
            **selected_metrics,
            "prediction_counts": {
                "0": int(selected_counts[0]),
                "1": int(selected_counts[1]),
            },
            "context_metrics": subgroup_metrics(truth, selected_predictions, contexts),
        },
        "macro_f1_gain_over_050": float(selected_metrics["macro_f1"] - metrics_050["macro_f1"]),
        "seed_metrics": seed_metrics,
        "selected_threshold_seed_macro_f1": {
            "mean": float(np.mean(seed_scores)),
            "std": float(np.std(seed_scores, ddof=0)),
            "minimum": float(np.min(seed_scores)),
            "maximum": float(np.max(seed_scores)),
        },
        "candidate_maximum_predicted_class_share_at_050": candidate_share,
        "candidate_accepted_at_050": candidate_share <= 0.90,
    }
    if corrected:
        threshold_report["route_threshold_diagnostics"] = route_threshold_diagnostics(
            truth,
            mean_probabilities,
            contexts,
        )
    reproduction_config = {
        "mode": args.mode,
        "candidate": candidate_name,
        "model_snapshot": MODEL_SNAPSHOT,
        "model_path_role": "repository_ignored_authenticated_snapshot",
        "seeds": list(VALIDATION_SEEDS),
        "folds_per_seed": FOLD_COUNT,
        "expected_fits": len(VALIDATION_SEEDS) * FOLD_COUNT,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "batch_size": config.batch_size,
        "gradient_accumulation": config.gradient_accumulation,
        "maximum_length": config.maximum_length,
        "precision": "cuda_fp16",
        "dataloader_workers": 0,
        "early_stopping": "none",
        "checkpoint_selection": (
            "final_epoch_historical_v3_cv" if historical else config.checkpoint_policy
        ),
        "checkpoint_retention": (
            "representative_seed_17_fold_1_only"
            if historical or corrected
            else "all_selected_fold_checkpoints"
        ),
        "threshold_grid": list(config.threshold_grid),
        "git_commit": _git_commit(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    runtime = {
        "wall_clock_seconds": perf_counter() - started,
        "maximum_fold_peak_gpu_vram_bytes": maximum_peak,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_total_bytes": checkpoint_total_bytes,
        "retained_checkpoint_bytes": retained_checkpoint_bytes,
    }
    _write_json(output_path / "config.json", reproduction_config)
    _write_json(output_path / "fold_metrics.json", {"folds": fold_records})
    _write_json(
        output_path / "aggregate_metrics.json",
        {
            "fixed_threshold_fold_summary": summarize_fold_metrics(fold_records),
            "threshold_selection": threshold_report,
        },
    )
    _write_json(output_path / "truncation.json", truncation_summary)
    if route_audit is not None:
        _write_json(output_path / "route_audit.json", route_audit)
    _write_json(output_path / "runtime.json", runtime)
    _write_json(
        output_path / "log.json",
        {
            "status": "passed" if candidate_share <= 0.90 else "failed_class_collapse",
            "official_rows_loaded": len(frame),
            "model_revision": model_loading["authentication"]["revision"],
            "weight_sha256": model_loading["authentication"]["weight_sha256"],
            "competition_test_accessed": False,
            "public_data_used": False,
            "row_level_probabilities_persisted": False,
        },
    )
    return {
        "status": "passed" if candidate_share <= 0.90 else "failed_class_collapse",
        "fits": len(fold_records),
        "threshold_selection": threshold_report,
        "runtime": runtime,
    }


def main(argv: list[str] | None = None) -> int:
    """Parse, validate, and run one explicitly bounded CLI mode."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    try:
        if args.mode == "smoke":
            result = run_smoke(args, repository_root())
        elif args.mode == "diagnose-reproduction":
            result = run_reproduction_diagnostics(args, repository_root())
        else:
            result = run_reproduction(args, repository_root())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
