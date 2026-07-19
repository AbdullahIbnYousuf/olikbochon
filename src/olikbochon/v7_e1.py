"""Common-fold E1: exact context rule plus Candidate I sparse null-route model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .metrics import classification_metrics, predictions_from_label1
from .v5_lexical import build_v5_groups, deterministic_substring_prediction
from .v5_sparse import CANDIDATES, SparseNullModel
from .v7_protocol import (
    FitScope,
    OUTER_FOLDS,
    OUTER_SEEDS,
    SelectionAudit,
    build_common_fold_assignments,
    calibration_table,
    has_context,
    make_inner_grouped_folds,
    metric_record,
    outer_indices,
    stable_training_row_ids,
    validate_probability_frame,
)


EXPERIMENT = "e1_exact_rule_candidate_i"
CANDIDATE_I = CANDIDATES[4]
C_GRID = (0.25, 1.0, 4.0)
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(40, 61, 2))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _exact_probabilities(frame: pd.DataFrame, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in frame.iloc[list(indices)].itertuples(index=False)
        ],
        dtype=np.float64,
    )


def _candidate_i_probabilities(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    presence: np.ndarray,
    *,
    classifier_c: float,
) -> tuple[np.ndarray, SparseNullModel]:
    train_absent = train_indices[~presence[train_indices]]
    validation_absent = validation_indices[~presence[validation_indices]]
    model = SparseNullModel(CANDIDATE_I, classifier_c=classifier_c).fit(
        frame.iloc[train_absent].reset_index(drop=True)
    )
    probabilities = model.predict_label1_probability(
        frame.iloc[validation_absent].reset_index(drop=True)
    )
    return probabilities, model


def select_inner_configuration(
    frame: pd.DataFrame,
    labels: np.ndarray,
    row_ids: np.ndarray,
    groups: np.ndarray,
    presence: np.ndarray,
    outer_train: np.ndarray,
    *,
    seed: int,
    outer_fold: int,
) -> tuple[float, float, dict[str, Any], np.ndarray]:
    """Select C and decision threshold without seeing the outer validation fold."""
    inner_folds = make_inner_grouped_folds(
        labels, groups, outer_train, seed=seed, outer_fold=outer_fold
    )
    candidates: list[tuple[tuple[float, ...], float, float, dict[str, Any]]] = []
    probabilities_by_c: dict[float, np.ndarray] = {}
    for classifier_c in C_GRID:
        probabilities = np.full(len(frame), np.nan, dtype=np.float64)
        for split in inner_folds:
            inner_train = np.asarray(split.train_indices, dtype=np.int64)
            inner_validation = np.asarray(split.validation_indices, dtype=np.int64)
            FitScope(
                role="e1_inner_vectorizer_scaler_classifier_fit",
                training_row_ids=tuple(row_ids[inner_train]),
                validation_row_ids=tuple(row_ids[inner_validation]),
            ).validate()
            for selection_kind in ("model_hyperparameter", "classification_threshold"):
                SelectionAudit(
                    selection_kind=selection_kind,
                    inner_training_row_ids=tuple(row_ids[inner_train]),
                    inner_validation_row_ids=tuple(row_ids[inner_validation]),
                ).validate()
            present_validation = inner_validation[presence[inner_validation]]
            probabilities[present_validation] = _exact_probabilities(
                frame, present_validation
            )
            absent_probabilities, model = _candidate_i_probabilities(
                frame,
                inner_train,
                inner_validation,
                presence,
                classifier_c=classifier_c,
            )
            absent_validation = inner_validation[~presence[inner_validation]]
            probabilities[absent_validation] = absent_probabilities
            if model.fit_audit()["training_row_count"] != int((~presence[inner_train]).sum()):
                raise RuntimeError("Candidate I fit scope does not match inner training rows")
        selected_probabilities = probabilities[outer_train]
        if not np.isfinite(selected_probabilities).all():
            raise RuntimeError("Inner OOF probabilities do not cover outer-training rows")
        probabilities_by_c[classifier_c] = selected_probabilities.copy()
        for threshold in THRESHOLD_GRID:
            predictions = predictions_from_label1(selected_probabilities, threshold)
            metrics = classification_metrics(labels[outer_train], predictions)
            counts = np.bincount(predictions, minlength=2)
            maximum_share = float(counts.max() / counts.sum())
            if maximum_share > 0.90:
                continue
            record = {
                "classifier_c": classifier_c,
                "classification_threshold": threshold,
                **metrics,
                "maximum_predicted_class_share": maximum_share,
                "selection_scope": "outer_training_inner_grouped_oof_only",
            }
            rank = (
                float(metrics["macro_f1"]),
                float(metrics["f1_label0"]),
                -abs(float(threshold) - 0.5),
                -abs(float(np.log10(classifier_c))),
                -float(threshold),
            )
            candidates.append((rank, classifier_c, threshold, record))
    if not candidates:
        raise RuntimeError("Every predeclared E1 inner configuration collapsed")
    _, classifier_c, threshold, record = max(candidates, key=lambda item: item[0])
    return (
        float(classifier_c),
        float(threshold),
        record,
        probabilities_by_c[float(classifier_c)],
    )


def run_e1(train_path: Path, output_dir: Path, folds_path: Path) -> dict[str, Any]:
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"E1 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    folds_path.parent.mkdir(parents=True, exist_ok=True)

    frame = load_labeled_json(train_path, expected_sha256=OFFICIAL_SAMPLE_SHA256).reset_index(
        drop=True
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    presence = np.asarray([has_context(value) for value in frame["context"]], dtype=bool)
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    group_audit = build_v5_groups(frame)
    groups = np.asarray(group_audit.group_ids, dtype=object)
    assignments = build_common_fold_assignments(frame)
    assignments.to_csv(folds_path, index=False)

    oof_records: list[pd.DataFrame] = []
    stack_training_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            fold_started = perf_counter()
            outer_train, outer_validation = outer_indices(
                assignments, row_ids, seed=seed, outer_fold=outer_fold
            )
            if set(groups[outer_train]) & set(groups[outer_validation]):
                raise RuntimeError("Outer grouped split leaks a group")
            FitScope(
                role="e1_outer_model_fit",
                training_row_ids=tuple(row_ids[outer_train]),
                validation_row_ids=tuple(row_ids[outer_validation]),
            ).validate()
            classifier_c, threshold, inner_record, inner_probabilities = select_inner_configuration(
                frame,
                labels,
                row_ids,
                groups,
                presence,
                outer_train,
                seed=seed,
                outer_fold=outer_fold,
            )
            stack_training_records.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[outer_train],
                        "seed": seed,
                        "target_outer_fold": outer_fold,
                        "label": labels[outer_train],
                        "context_present": presence[outer_train],
                        "probability_label_1_e1": inner_probabilities,
                        "probability_label_0_e1": 1.0 - inner_probabilities,
                        "classification_threshold_e1": threshold,
                        "base_prediction_provenance": "outer_train_inner_grouped_cross_fit",
                    }
                )
            )
            probabilities = np.full(len(outer_validation), np.nan, dtype=np.float64)
            present_local = presence[outer_validation]
            probabilities[present_local] = _exact_probabilities(
                frame, outer_validation[present_local]
            )
            absent_probabilities, model = _candidate_i_probabilities(
                frame,
                outer_train,
                outer_validation,
                presence,
                classifier_c=classifier_c,
            )
            probabilities[~present_local] = absent_probabilities
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Outer validation probabilities are incomplete")

            model_path = models_dir / f"seed_{seed}_fold_{outer_fold}.joblib"
            joblib.dump(model, model_path, compress=3)
            reloaded = joblib.load(model_path)
            absent_frame = frame.iloc[outer_validation[~present_local]].reset_index(drop=True)
            reloaded_probabilities = reloaded.predict_label1_probability(absent_frame)
            reload_max_abs_diff = float(
                np.max(np.abs(reloaded_probabilities - absent_probabilities))
                if len(absent_probabilities)
                else 0.0
            )
            if reload_max_abs_diff > 1e-12:
                raise RuntimeError("Reloaded Candidate I predictions differ")

            predictions = predictions_from_label1(probabilities, threshold)
            record = {
                "seed": seed,
                "outer_fold": outer_fold,
                "train_rows": int(len(outer_train)),
                "validation_rows": int(len(outer_validation)),
                "train_groups": int(len(set(groups[outer_train]))),
                "validation_groups": int(len(set(groups[outer_validation]))),
                "group_overlap_count": 0,
                "classifier_c": classifier_c,
                "classification_threshold": threshold,
                "inner_selection": inner_record,
                "runtime_seconds": perf_counter() - fold_started,
                "peak_vram_bytes": None,
                "reload_max_abs_probability_difference": reload_max_abs_diff,
                **metric_record(
                    labels[outer_validation],
                    probabilities,
                    predictions,
                    presence[outer_validation],
                ),
            }
            fold_metrics.append(record)
            oof_records.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[outer_validation],
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "label": labels[outer_validation],
                        "context_present": presence[outer_validation],
                        "probability_label_0": 1.0 - probabilities,
                        "probability_label_1": probabilities,
                        "classification_threshold": threshold,
                        "prediction": predictions,
                    }
                )
            )
            model_manifest.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "path": model_path.relative_to(output_dir).as_posix(),
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                    "classifier_c": classifier_c,
                    "probability_orientation": "probability_label_1_is_faithful",
                    "reload_max_abs_probability_difference": reload_max_abs_diff,
                }
            )

    oof = pd.concat(oof_records, ignore_index=True).sort_values(
        ["seed", "outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_probability_frame(oof, row_ids)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    stack_training = pd.concat(stack_training_records, ignore_index=True).sort_values(
        ["seed", "target_outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    stack_training.to_csv(output_dir / "stack_training_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).drop(columns=["route_metrics", "inner_selection"]).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )

    seed_records: list[dict[str, Any]] = []
    for seed, selected in oof.groupby("seed", sort=True):
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
    aggregate = metric_record(
        oof["label"],
        oof["probability_label_1"],
        oof["prediction"],
        oof["context_present"],
    )
    summary = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "probability_orientation": "probability_label_1_is_faithful_label_1",
        "official_rows": int(len(frame)),
        "oof_rows": int(len(oof)),
        "complete_oof_predictions_per_official_row": len(OUTER_SEEDS),
        "group_audit": group_audit.__dict__,
        "aggregate_repeated_oof": aggregate,
        "seed_metrics": seed_records,
        "fold_metric_distribution": {
            name: {
                "mean": float(np.mean([record[name] for record in fold_metrics])),
                "std": float(np.std([record[name] for record in fold_metrics], ddof=0)),
                "minimum": float(np.min([record[name] for record in fold_metrics])),
                "maximum": float(np.max([record[name] for record in fold_metrics])),
            }
            for name in ("macro_f1", "f1_label0", "f1_label1", "brier_score")
        },
        "calibration": calibration_table(oof["label"], oof["probability_label_1"]),
        "runtime_seconds": perf_counter() - started,
        "peak_vram_bytes": None,
        "test_rows_accessed": 0,
        "reload_all_equal": True,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "confusion_matrices.json",
        {
            "aggregate": aggregate["confusion_matrix"],
            "routes": {
                name: values["confusion_matrix"]
                for name, values in aggregate["route_metrics"].items()
            },
            "folds": [
                {
                    "seed": row["seed"],
                    "outer_fold": row["outer_fold"],
                    "confusion_matrix": row["confusion_matrix"],
                }
                for row in fold_metrics
            ],
        },
    )
    _write_json(
        output_dir / "config.json",
        {
            "experiment": EXPERIMENT,
            "seeds": OUTER_SEEDS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": 3,
            "candidate": CANDIDATE_I,
            "candidate_i_channels": ["combined_char", "combined_word", "numeric"],
            "classifier_c_grid": C_GRID,
            "classification_threshold_grid": THRESHOLD_GRID,
            "context_present_model": "deterministic_normalized_exact_substring",
            "selection_scope": "nested_inner_grouped_only",
            "test_data_role": "not_accessed",
        },
    )
    _write_json(output_dir / "model_manifest.json", model_manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7/e1"))
    parser.add_argument(
        "--folds-path", type=Path, default=Path("artifacts/v7/folds/common_folds.csv")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_e1(args.train_data, args.output_dir, args.folds_path)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "experiment": summary["experiment"],
                "macro_f1": summary["aggregate_repeated_oof"]["macro_f1"],
                "test_rows_accessed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
