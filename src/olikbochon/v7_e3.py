"""Nested deterministic route blend for compatible common-fold E0 and E1 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import DataValidationError
from .metrics import classification_metrics, predictions_from_label1
from .v7_e1 import _write_json
from .v7_protocol import (
    OUTER_FOLDS,
    OUTER_SEEDS,
    calibration_table,
    metric_record,
    validate_probability_frame,
)


EXPERIMENT = "e3_deterministic_route_blend"
RULES = (
    "e0_all_routes",
    "e1_all_routes",
    "e1_present_e0_absent",
    "e1_present_equal_absent_blend",
    "e1_present_e1_75_absent_blend",
)
THRESHOLDS = tuple(round(value / 100, 2) for value in range(40, 61, 2))


def route_rule_probability(
    rule: str,
    probability_e0: np.ndarray,
    probability_e1: np.ndarray,
    context_present: np.ndarray,
) -> np.ndarray:
    if rule not in RULES:
        raise ValueError(rule)
    e0 = np.asarray(probability_e0, dtype=np.float64)
    e1 = np.asarray(probability_e1, dtype=np.float64)
    present = np.asarray(context_present, dtype=bool)
    if not (e0.shape == e1.shape == present.shape):
        raise DataValidationError("E3 rule inputs must be aligned")
    if rule == "e0_all_routes":
        return e0.copy()
    if rule == "e1_all_routes":
        return e1.copy()
    result = e1.copy()
    if rule == "e1_present_e0_absent":
        result[~present] = e0[~present]
    elif rule == "e1_present_equal_absent_blend":
        result[~present] = 0.5 * e0[~present] + 0.5 * e1[~present]
    else:
        result[~present] = 0.25 * e0[~present] + 0.75 * e1[~present]
    return result


def select_rule(
    labels: np.ndarray,
    probability_e0: np.ndarray,
    probability_e1: np.ndarray,
    context_present: np.ndarray,
) -> tuple[str, float, dict[str, Any]]:
    ranked: list[tuple[tuple[float, ...], str, float, dict[str, Any]]] = []
    for rule in RULES:
        probabilities = route_rule_probability(
            rule, probability_e0, probability_e1, context_present
        )
        for threshold in THRESHOLDS:
            predictions = predictions_from_label1(probabilities, threshold)
            metrics = classification_metrics(labels, predictions)
            counts = np.bincount(predictions, minlength=2)
            share = float(counts.max() / counts.sum())
            if share > 0.90:
                continue
            record = {
                "rule": rule,
                "classification_threshold": threshold,
                **metrics,
                "maximum_predicted_class_share": share,
                "selection_scope": "outer_training_inner_grouped_cross_fitted_bases_only",
            }
            rank = (
                metrics["macro_f1"],
                metrics["f1_label0"],
                -abs(threshold - 0.5),
                -RULES.index(rule),
                -threshold,
            )
            ranked.append((rank, rule, threshold, record))
    if not ranked:
        raise RuntimeError("Every predeclared E3 rule collapsed")
    _, rule, threshold, record = max(ranked, key=lambda item: item[0])
    return rule, float(threshold), record


def _aligned_merge(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise DataValidationError("E3 inputs contain duplicate stable keys")
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise DataValidationError("E3 inputs do not have identical stable-key coverage")
    return merged


def run_e3(e0_dir: Path, e1_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for required in (
        e0_dir / "oof_predictions.csv",
        e0_dir / "stack_training_predictions.csv",
        e1_dir / "oof_predictions.csv",
        e1_dir / "stack_training_predictions.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"E3 requires regenerated compatible-fold base artifact: {required}"
            )
    output_dir.mkdir(parents=True)
    e0_oof = pd.read_csv(e0_dir / "oof_predictions.csv").rename(
        columns={"probability_label_1": "probability_label_1_e0"}
    )
    e1_oof = pd.read_csv(e1_dir / "oof_predictions.csv").rename(
        columns={"probability_label_1": "probability_label_1_e1"}
    )
    keys = ["row_id", "seed", "outer_fold", "label", "context_present"]
    oof = _aligned_merge(
        e0_oof[keys + ["probability_label_1_e0"]],
        e1_oof[keys + ["probability_label_1_e1"]],
        keys,
    )
    e0_stack = pd.read_csv(e0_dir / "stack_training_predictions.csv")
    e1_stack = pd.read_csv(e1_dir / "stack_training_predictions.csv")
    stack_keys = [
        "row_id",
        "seed",
        "target_outer_fold",
        "label",
        "context_present",
        "base_prediction_provenance",
    ]
    stack = _aligned_merge(
        e0_stack[stack_keys + ["probability_label_1_e0"]],
        e1_stack[stack_keys + ["probability_label_1_e1"]],
        stack_keys,
    )
    if set(stack["base_prediction_provenance"]) != {
        "outer_train_inner_grouped_cross_fit"
    }:
        raise DataValidationError("E3 base predictions are not inner cross-fitted")

    result_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            fold_started = perf_counter()
            training = stack.loc[
                (stack["seed"] == seed)
                & (stack["target_outer_fold"] == outer_fold)
            ]
            validation = oof.loc[
                (oof["seed"] == seed) & (oof["outer_fold"] == outer_fold)
            ].copy()
            if training.empty or validation.empty:
                raise DataValidationError("E3 fold inputs are incomplete")
            if set(training["row_id"]) & set(validation["row_id"]):
                raise DataValidationError("E3 selection sees outer-validation IDs")
            rule, threshold, selection = select_rule(
                training["label"].to_numpy(dtype=np.int64),
                training["probability_label_1_e0"].to_numpy(dtype=np.float64),
                training["probability_label_1_e1"].to_numpy(dtype=np.float64),
                training["context_present"].to_numpy(dtype=bool),
            )
            probabilities = route_rule_probability(
                rule,
                validation["probability_label_1_e0"].to_numpy(dtype=np.float64),
                validation["probability_label_1_e1"].to_numpy(dtype=np.float64),
                validation["context_present"].to_numpy(dtype=bool),
            )
            predictions = predictions_from_label1(probabilities, threshold)
            validation["probability_label_0"] = 1.0 - probabilities
            validation["probability_label_1"] = probabilities
            validation["classification_threshold"] = threshold
            validation["selected_rule"] = rule
            validation["prediction"] = predictions
            result_parts.append(validation)
            fold_records.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "selected_rule": rule,
                    "classification_threshold": threshold,
                    "inner_selection": selection,
                    "runtime_seconds": perf_counter() - fold_started,
                    "peak_vram_bytes": None,
                    **metric_record(
                        validation["label"],
                        probabilities,
                        predictions,
                        validation["context_present"],
                    ),
                }
            )
    result = pd.concat(result_parts, ignore_index=True).sort_values(
        ["seed", "outer_fold", "row_id"], kind="mergesort"
    )
    expected_ids = result.loc[result["seed"] == OUTER_SEEDS[0], "row_id"].astype(str)
    validate_probability_frame(result, expected_ids)
    result.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_records).drop(columns=["inner_selection", "route_metrics"]).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    aggregate = metric_record(
        result["label"],
        result["probability_label_1"],
        result["prediction"],
        result["context_present"],
    )
    seed_records: list[dict[str, Any]] = []
    for seed, selected in result.groupby("seed", sort=True):
        seed_records.append(
            {
                "seed": int(seed),
                **metric_record(
                    selected["label"],
                    selected["probability_label_1"],
                    selected["prediction"],
                    selected["context_present"],
                ),
            }
        )
    pd.DataFrame(seed_records).drop(columns=["route_metrics"]).to_csv(
        output_dir / "seed_metrics.csv", index=False
    )
    pd.DataFrame(
        [{"route": name, **values} for name, values in aggregate["route_metrics"].items()]
    ).to_csv(output_dir / "route_metrics.csv", index=False)
    summary = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "aggregate_repeated_oof": aggregate,
        "seed_metrics": seed_records,
        "calibration": calibration_table(result["label"], result["probability_label_1"]),
        "runtime_seconds": perf_counter() - started,
        "peak_vram_bytes": None,
        "reload_equality": "not_applicable_deterministic_probability_rule",
        "test_rows_accessed": 0,
        "base_prediction_provenance": "outer_train_inner_grouped_cross_fit",
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "config.json",
        {
            "experiment": EXPERIMENT,
            "rules": RULES,
            "thresholds": THRESHOLDS,
            "probability_orientation": "probability_label_1_is_faithful_label_1",
            "selection_scope": "outer_training_inner_grouped_cross_fitted_bases_only",
            "test_data_role": "not_accessed",
        },
    )
    _write_json(output_dir / "confusion_matrices.json", {"aggregate": aggregate})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0-dir", type=Path, default=Path("artifacts/v7/e0"))
    parser.add_argument("--e1-dir", type=Path, default=Path("artifacts/v7/e1"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7/e3"))
    args = parser.parse_args()
    summary = run_e3(args.e0_dir, args.e1_dir, args.output_dir)
    print(json.dumps({"status": summary["status"], "experiment": EXPERIMENT}))


if __name__ == "__main__":
    main()
