"""Constrained CLI for the frozen official-only V6 null-route neural experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .v4_runner import load_official_training_frame, require_approved_model_path
from .v4_validation import FOLD_COUNT, THRESHOLD_GRID, VALIDATION_SEEDS
from .v6_null_neural import (
    BATCH_SIZE,
    CANDIDATES,
    CANDIDATE_SET,
    EFFECTIVE_BATCH_SIZE,
    EXPERIMENT_NAME,
    GRADIENT_ACCUMULATION,
    MAXIMUM_EPOCHS,
    MAXIMUM_LENGTH,
    POLICIES,
    run_v6_experiment,
)


MODE = "null-neural-baseline"
OUTPUT_NAME = "null_neural_baseline"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frozen_config() -> dict[str, Any]:
    """Return the complete protocol before any V6 neural result is observed."""
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": MODE,
        "candidate_set": CANDIDATE_SET,
        "candidates": list(CANDIDATES),
        "policies": {
            candidate: {
                "encoder_policy": policy.encoder_policy,
                "head_learning_rate": policy.head_learning_rate,
                "encoder_learning_rate": policy.encoder_learning_rate,
            }
            for candidate, policy in POLICIES.items()
        },
        "data_role": "authenticated_official_labeled_sample_only",
        "fit_route": "corrected_context_absent_only",
        "present_route": "frozen_candidate_a_exact_normalized_substring_rule",
        "serialization": "[QUESTION] newline normalized prompt blank-line [ANSWER] newline normalized response",
        "maximum_length": MAXIMUM_LENGTH,
        "epochs": MAXIMUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "precision": "CUDA fp16",
        "workers": 0,
        "class_weights": "balanced weights fitted on current null-route training fold only",
        "checkpoint_selection": "validation macro F1 at 0.50, then validation loss, then earlier epoch",
        "thresholds": {
            "primary": 0.50,
            "global_pooled_grid": list(THRESHOLD_GRID),
            "maximum_predicted_class_share": 0.90,
        },
        "retention": "reload all selected deltas; retain only seed 17 fold 1 per candidate",
        "artifact_contract": "aggregate/fold numeric metadata and checkpoints only; no raw text or row-level values",
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v6_runner",
        description="Run the one frozen official-only V6 null-route neural experiment.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--candidate-set", required=True, choices=(CANDIDATE_SET,))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def require_v6_output_path(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    rules = {
        line.strip()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    if "artifacts/v6/" not in rules:
        raise ValueError("artifacts/v6/ must be explicitly ignored")
    if not output_path.is_absolute():
        raise ValueError("V6 output directory must be an absolute path")
    output = output_path.resolve()
    expected = (root / "artifacts" / "v6" / OUTPUT_NAME).resolve()
    if output != expected:
        raise ValueError(f"V6 output must be artifacts/v6/{OUTPUT_NAME}")
    if output.exists():
        raise ValueError("V6 output directory must not already exist")
    return output


def validate_arguments(args: argparse.Namespace, root: Path) -> tuple[Path, Path]:
    if args.mode != MODE or args.candidate_set != CANDIDATE_SET:
        raise ValueError("V6 requires the frozen null-neural candidate set")
    if tuple(args.seeds) != VALIDATION_SEEDS:
        raise ValueError("V6 requires --seeds 17 29 43 in that order")
    if args.folds != FOLD_COUNT:
        raise ValueError("V6 requires --folds 5")
    if args.max_length != MAXIMUM_LENGTH:
        raise ValueError("V6 requires --max-length 192")
    return (
        require_approved_model_path(root, args.model_path),
        require_v6_output_path(root, args.output_dir),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    root = repository_root()
    model_path, output = validate_arguments(args, root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    frame = load_official_training_frame(root)
    result = run_v6_experiment(frame, model_path, output)
    (output / "frozen_config.json").write_text(
        json.dumps(frozen_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "status": "complete",
                "total_fits": result["total_fits"],
                "output": str(output.relative_to(root)),
                "raw_text_persisted": False,
                "row_level_values_persisted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
