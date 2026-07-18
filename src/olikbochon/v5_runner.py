"""Constrained official-only CLI for the frozen V5 lexical baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .v4_runner import load_official_training_frame
from .v4_validation import FOLD_COUNT, THRESHOLD_GRID, VALIDATION_SEEDS
from .v5_features import CONTEXT_ABSENT_FEATURES, CONTEXT_PRESENT_FEATURES
from .v5_lexical import (
    CANDIDATES,
    CANDIDATE_SET,
    EXPERIMENT_NAME,
    LOGISTIC_C,
    LOGISTIC_MAX_ITERATIONS,
    LOGISTIC_RANDOM_STATE,
    LOGISTIC_SOLVER,
    run_frozen_lexical_experiment,
)


MODE = "lexical-baseline"
OUTPUT_NAME = "official_lexical_baseline"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frozen_config() -> dict[str, Any]:
    """Return the complete predeclared configuration before any candidate is scored."""
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": MODE,
        "candidate_set": CANDIDATE_SET,
        "candidates": list(CANDIDATES),
        "data_role": "authenticated_official_labeled_sample_only",
        "validation": {
            "seeds": list(VALIDATION_SEEDS),
            "folds": FOLD_COUNT,
            "strategy": "repeated_stratified_group_kfold_no_random_fallback",
            "groups": [
                "v5_normalized_exact_row",
                "v5_normalized_prompt_identity",
                "v5_normalized_present_context_identity",
                "v3_family_groups",
                "v3_near_duplicate_groups",
            ],
        },
        "context_present_features": list(CONTEXT_PRESENT_FEATURES),
        "context_absent_features": list(CONTEXT_ABSENT_FEATURES),
        "candidate_a_absent_fallback": (
            "training_fold_context_absent_majority; exact ties predict label 0"
        ),
        "logistic": {
            "standard_scaler": True,
            "class_weight": "balanced",
            "C": LOGISTIC_C,
            "max_iter": LOGISTIC_MAX_ITERATIONS,
            "random_state": LOGISTIC_RANDOM_STATE,
            "solver": LOGISTIC_SOLVER,
        },
        "thresholds": {
            "primary": 0.50,
            "secondary_grid": list(THRESHOLD_GRID),
            "maximum_predicted_class_share": 0.90,
        },
        "acceptance": {
            "substring_promising_context_macro_f1": 0.75,
            "substring_strong_context_macro_f1": 0.85,
            "present_logistic_gain_over_substring": 0.01,
            "absent_logistic_pass_macro_f1": 0.55,
            "routed_strong_macro_f1": 0.62,
            "routed_high_value_macro_f1": 0.68,
            "routed_reject_below_macro_f1": 0.565096,
        },
        "artifact_contract": (
            "aggregate and fold metrics, aggregate feature statistics, standardized "
            "coefficients, and frozen config only; no text and no row-level table"
        ),
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v5_runner",
        description="Run the single frozen official-only V5 lexical experiment.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--candidate-set", required=True, choices=(CANDIDATE_SET,))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def require_v5_output_path(root: Path, output_path: Path) -> Path:
    """Require a new exact output below an explicitly ignored V5 artifact root."""
    root = Path(root).resolve()
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file() or "artifacts/v5/" not in ignore_file.read_text(
        encoding="utf-8"
    ).splitlines():
        raise ValueError("artifacts/v5/ must be explicitly ignored")
    if not output_path.is_absolute():
        raise ValueError("V5 output directory must be an absolute path")
    output = output_path.resolve()
    expected = (root / "artifacts" / "v5" / OUTPUT_NAME).resolve()
    if output != expected:
        raise ValueError(f"V5 output must be artifacts/v5/{OUTPUT_NAME}")
    if output.exists():
        raise ValueError("V5 output directory must not already exist")
    return output


def validate_arguments(args: argparse.Namespace, root: Path) -> Path:
    if args.mode != MODE or args.candidate_set != CANDIDATE_SET:
        raise ValueError("V5 requires the frozen lexical-baseline candidate set")
    if tuple(args.seeds) != VALIDATION_SEEDS:
        raise ValueError("V5 requires --seeds 17 29 43 in that order")
    if args.folds != FOLD_COUNT:
        raise ValueError("V5 requires --folds 5")
    return require_v5_output_path(root, args.output_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    root = repository_root()
    output = validate_arguments(args, root)
    frame = load_official_training_frame(root)
    result = run_frozen_lexical_experiment(frame)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "frozen_config.json", frozen_config())
    _write_json(output / "results.json", result)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "output": str(output.relative_to(root)),
                "row_level_values_persisted": False,
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
