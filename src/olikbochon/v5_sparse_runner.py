"""Constrained CLI for the frozen official-only V5 null-route sparse experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .v4_runner import load_official_training_frame
from .v4_validation import FOLD_COUNT, THRESHOLD_GRID, VALIDATION_SEEDS
from .v5_features import CONTEXT_ABSENT_FEATURES
from .v5_sparse import (
    CANDIDATES,
    CANDIDATE_SET,
    CHAR_SPEC,
    EXPERIMENT_NAME,
    LOGISTIC_CONFIG,
    PROMPT_WORD_SPEC,
    RESPONSE_CHAR_SPEC,
    WORD_SPEC,
    run_sparse_experiment,
)


MODE = "null-sparse-baseline"
OUTPUT_NAME = "null_sparse_baseline"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def frozen_config() -> dict[str, Any]:
    """Return the complete design before any official sparse candidate is evaluated."""
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": MODE,
        "candidate_set": CANDIDATE_SET,
        "candidates": list(CANDIDATES),
        "data_role": "authenticated_official_labeled_sample_only",
        "fit_route": "corrected_context_absent_only",
        "present_route": "frozen_candidate_a_exact_normalized_substring_rule",
        "views": {
            "view_1": "[QUESTION] normalized prompt newline [ANSWER] normalized response",
            "view_2": "independently vectorized normalized prompt and response",
            "view_3": (
                "view_1 character+word union plus the frozen 21 absent-route features"
            ),
        },
        "vectorizers": {
            "combined_character": asdict(CHAR_SPEC),
            "combined_word": asdict(WORD_SPEC),
            "separate_prompt_word": asdict(PROMPT_WORD_SPEC),
            "separate_response_character": asdict(RESPONSE_CHAR_SPEC),
        },
        "candidate_i_numeric_features": list(CONTEXT_ABSENT_FEATURES),
        "candidate_i_numeric_scaler": "StandardScaler fitted on current training fold only",
        "classifier": LOGISTIC_CONFIG,
        "validation": {
            "seeds": list(VALIDATION_SEEDS),
            "folds": FOLD_COUNT,
            "groups": [
                "v5_normalized_exact_row",
                "v5_normalized_prompt_identity",
                "v5_normalized_present_context_identity",
                "v3_family_groups",
                "v3_near_duplicate_groups",
            ],
            "random_fallback": False,
        },
        "thresholds": {
            "primary": 0.50,
            "null_route_secondary_grid": list(THRESHOLD_GRID),
            "maximum_predicted_class_share": 0.90,
        },
        "acceptance": {
            "null_weak_below": 0.55,
            "null_promising_at_least": 0.58,
            "null_strong_at_least": 0.62,
            "null_high_value_at_least": 0.68,
            "routed_champion_to_beat": 0.683111,
            "routed_meaningful_gain": 0.01,
            "routed_strong_at_least": 0.70,
            "routed_high_value_at_least": 0.74,
        },
        "artifact_contract": (
            "frozen config plus aggregate/fold metrics and vocabulary sizes only; "
            "no raw text, vocabulary terms, or row-level values"
        ),
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v5_sparse_runner",
        description="Run the one frozen official-only null-route sparse experiment.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--candidate-set", required=True, choices=(CANDIDATE_SET,))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def require_sparse_output_path(root: Path, output_path: Path) -> Path:
    """Require the exact new ignored sparse output and reject all input-like paths."""
    root = Path(root).resolve()
    ignore_file = root / ".gitignore"
    if not ignore_file.is_file() or "artifacts/v5/" not in ignore_file.read_text(
        encoding="utf-8"
    ).splitlines():
        raise ValueError("artifacts/v5/ must be explicitly ignored")
    if not output_path.is_absolute():
        raise ValueError("Sparse output directory must be an absolute path")
    output = output_path.resolve()
    expected = (root / "artifacts" / "v5" / OUTPUT_NAME).resolve()
    if output != expected:
        raise ValueError(f"Sparse output must be artifacts/v5/{OUTPUT_NAME}")
    if output.exists():
        raise ValueError("Sparse output directory must not already exist")
    return output


def validate_arguments(args: argparse.Namespace, root: Path) -> Path:
    if args.mode != MODE or args.candidate_set != CANDIDATE_SET:
        raise ValueError("Sparse V5 requires the frozen null-sparse candidate set")
    if tuple(args.seeds) != VALIDATION_SEEDS:
        raise ValueError("Sparse V5 requires --seeds 17 29 43 in that order")
    if args.folds != FOLD_COUNT:
        raise ValueError("Sparse V5 requires --folds 5")
    return require_sparse_output_path(root, args.output_dir)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    root = repository_root()
    output = validate_arguments(args, root)
    frame = load_official_training_frame(root)
    result = run_sparse_experiment(frame)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "frozen_config.json", frozen_config())
    _write_json(output / "results.json", result)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "output": str(output.relative_to(root)),
                "row_level_values_persisted": False,
                "raw_vocabulary_persisted": False,
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
