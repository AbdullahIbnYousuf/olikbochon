"""Official-only Version 4 preparation and narrowly bounded smoke CLI."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json
from .metrics import predictions_from_label1, subgroup_metrics
from .v3_preprocessing import raw_context_is_present
from .v4_preprocessing import (
    V3_COMPATIBLE,
    EncodedV4Input,
    encode_comparison_baseline,
    encode_field_aware,
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
    require_local_model_path,
    train_smoke_model,
)
from .v4_validation import RepeatedGroupedFolds, build_repeated_official_folds


SMOKE_CANDIDATE = "v3_compatible_baseline"
SMOKE_SEED = 17
SMOKE_MAXIMUM_LENGTH = 256
SMOKE_BATCH_SIZE = 4
SMOKE_GRADIENT_ACCUMULATION = 2


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
    """Build the deliberately narrow Gate 4 smoke interface."""
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v4_runner",
        description="Run one bounded official-only V4 CUDA smoke experiment.",
    )
    parser.add_argument("--mode", required=True, choices=("smoke",))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--candidate", required=True, choices=(SMOKE_CANDIDATE,))
    parser.add_argument("--seed", required=True, type=int, choices=(SMOKE_SEED,))
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--max-length", required=True, type=int, choices=(SMOKE_MAXIMUM_LENGTH,))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_smoke_arguments(args: argparse.Namespace, root: Path) -> tuple[Path, Path]:
    """Validate local-only inputs and hard smoke bounds before any data or CUDA work."""
    if not 1 <= args.max_steps <= 20:
        raise ValueError("--max-steps must be between 1 and 20")
    model_path = require_approved_model_path(root, args.model_path)
    output_path = require_smoke_output_path(root, args.output_dir)
    return model_path, output_path


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


def main(argv: list[str] | None = None) -> int:
    """Parse, validate, and run the sole authorized CLI mode."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    try:
        result = run_smoke(args, repository_root())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
