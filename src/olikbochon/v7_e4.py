"""Common-fold E4: leakage-resistant nested logistic fusion of E0, E1, and E2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loading import OFFICIAL_SAMPLE_SHA256, DataValidationError, load_labeled_json, sha256_file
from .metrics import classification_metrics, predictions_from_label1
from .v5_lexical import build_v5_groups
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


EXPERIMENT = "e4_nested_logistic_stack_e0_e1_e2"
C_GRID = (0.1, 1.0, 10.0)
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(40, 61, 2))
FEATURE_COLUMNS = (
    "probability_label_1_e0",
    "probability_label_1_e1",
    "probability_label_1_e2",
    "context_present_numeric",
    "exact_match_numeric",
    "retrieval_score",
    "retrieval_accepted_numeric",
    "evidence_length_log1p",
)


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


def _read_base_training(directory: Path, suffix: str) -> pd.DataFrame:
    table = pd.read_csv(directory / "stack_training_predictions.csv")
    required = {
        "row_id",
        "seed",
        "target_outer_fold",
        "label",
        "context_present",
        f"probability_label_1_{suffix}",
        "base_prediction_provenance",
    }
    if not required.issubset(table.columns):
        raise DataValidationError(f"{suffix} stack-training artifact lacks required columns")
    if set(table["base_prediction_provenance"]) != {"outer_train_inner_grouped_cross_fit"}:
        raise DataValidationError(f"{suffix} base probabilities are not cross-fitted")
    key = ["row_id", "seed", "target_outer_fold"]
    if table.duplicated(key).any():
        raise DataValidationError(f"{suffix} stack-training artifact has duplicate keys")
    keep = key + ["label", "context_present", f"probability_label_1_{suffix}"]
    if suffix == "e2":
        keep += ["retrieval_score", "retrieval_accepted", "evidence_length"]
    return table[keep]


def _read_base_oof(directory: Path, suffix: str) -> pd.DataFrame:
    table = pd.read_csv(directory / "oof_predictions.csv")
    required = {
        "row_id", "seed", "outer_fold", "label", "context_present", "probability_label_1"
    }
    if not required.issubset(table.columns):
        raise DataValidationError(f"{suffix} OOF artifact lacks required columns")
    key = ["row_id", "seed"]
    if table.duplicated(key).any():
        raise DataValidationError(f"{suffix} OOF artifact has duplicate keys")
    rename = {"probability_label_1": f"probability_label_1_{suffix}"}
    keep = ["row_id", "seed", "outer_fold", "label", "context_present", "probability_label_1"]
    if suffix == "e2":
        keep += ["retrieval_score", "retrieval_accepted", "evidence_length"]
    return table[keep].rename(columns=rename)


def _merge_training(e0_dir: Path, e1_dir: Path, e2_dir: Path) -> pd.DataFrame:
    keys = ["row_id", "seed", "target_outer_fold"]
    e0 = _read_base_training(e0_dir, "e0")
    e1 = _read_base_training(e1_dir, "e1").drop(columns=["label", "context_present"])
    e2 = _read_base_training(e2_dir, "e2").drop(columns=["label", "context_present"])
    merged = e0.merge(e1, on=keys, how="inner", validate="one_to_one").merge(
        e2, on=keys, how="inner", validate="one_to_one"
    )
    if not (len(merged) == len(e0) == len(e1) == len(e2)):
        raise DataValidationError("Base stack-training matrices are not exactly aligned")
    return merged


def _merge_oof(e0_dir: Path, e1_dir: Path, e2_dir: Path) -> pd.DataFrame:
    keys = ["row_id", "seed", "outer_fold"]
    e0 = _read_base_oof(e0_dir, "e0")
    e1 = _read_base_oof(e1_dir, "e1").drop(columns=["label", "context_present"])
    e2 = _read_base_oof(e2_dir, "e2").drop(columns=["label", "context_present"])
    merged = e0.merge(e1, on=keys, how="inner", validate="one_to_one").merge(
        e2, on=keys, how="inner", validate="one_to_one"
    )
    if not (len(merged) == len(e0) == len(e1) == len(e2)):
        raise DataValidationError("Base OOF matrices are not exactly aligned")
    return merged


def add_meta_features(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    result["context_present_numeric"] = result["context_present"].astype(float)
    result["exact_match_numeric"] = (
        result["context_present"].astype(bool)
        & (result["probability_label_1_e1"] >= 0.5)
    ).astype(float)
    result["retrieval_accepted_numeric"] = result["retrieval_accepted"].astype(float)
    result["evidence_length_log1p"] = np.log1p(result["evidence_length"].astype(float))
    values = result[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataValidationError("E4 base-prediction features contain non-finite values")
    return result


def _fit_model(features: np.ndarray, labels: np.ndarray, classifier_c: float) -> dict[str, Any]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=float(classifier_c),
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    ).fit(scaler.transform(features), labels)
    return {"scaler": scaler, "classifier": classifier}


def _predict_model(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    classifier = model["classifier"]
    classes = list(classifier.classes_)
    return classifier.predict_proba(model["scaler"].transform(features))[:, classes.index(1)]


def select_inner_configuration(
    table: pd.DataFrame,
    labels: np.ndarray,
    row_ids: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    *,
    seed: int,
    outer_fold: int,
) -> tuple[float, float, dict[str, Any], np.ndarray]:
    lookup = {str(row_id): index for index, row_id in enumerate(row_ids)}
    table_indices = np.asarray([lookup[str(value)] for value in table["row_id"]], dtype=np.int64)
    if set(table_indices) != set(outer_train_indices):
        raise DataValidationError("E4 training base rows differ from the outer-training partition")
    row_to_local = {int(index): local for local, index in enumerate(table_indices)}
    features = table[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    table_labels = table["label"].to_numpy(dtype=np.int64)
    inner_folds = make_inner_grouped_folds(
        labels, groups, outer_train_indices, seed=seed, outer_fold=outer_fold
    )
    candidates: list[tuple[tuple[float, ...], float, float, dict[str, Any]]] = []
    probabilities_by_c: dict[float, np.ndarray] = {}
    for classifier_c in C_GRID:
        probabilities = np.full(len(table), np.nan, dtype=np.float64)
        for split in inner_folds:
            inner_train_ids = np.asarray(split.train_indices, dtype=np.int64)
            inner_validation_ids = np.asarray(split.validation_indices, dtype=np.int64)
            inner_train = np.asarray([row_to_local[int(value)] for value in inner_train_ids])
            inner_validation = np.asarray(
                [row_to_local[int(value)] for value in inner_validation_ids]
            )
            FitScope(
                role="e4_inner_meta_scaler_classifier_fit",
                training_row_ids=tuple(row_ids[inner_train_ids]),
                validation_row_ids=tuple(row_ids[inner_validation_ids]),
            ).validate()
            for selection_kind in ("fusion_regularization", "classification_threshold"):
                SelectionAudit(
                    selection_kind=selection_kind,
                    inner_training_row_ids=tuple(row_ids[inner_train_ids]),
                    inner_validation_row_ids=tuple(row_ids[inner_validation_ids]),
                ).validate()
            model = _fit_model(features[inner_train], table_labels[inner_train], classifier_c)
            probabilities[inner_validation] = _predict_model(model, features[inner_validation])
        if not np.isfinite(probabilities).all():
            raise RuntimeError("E4 nested meta cross-fit is incomplete")
        probabilities_by_c[float(classifier_c)] = probabilities.copy()
        for threshold in THRESHOLD_GRID:
            predictions = predictions_from_label1(probabilities, threshold)
            metrics = classification_metrics(table_labels, predictions)
            counts = np.bincount(predictions, minlength=2)
            maximum_share = float(counts.max() / counts.sum())
            if maximum_share > 0.90:
                continue
            record = {
                "classifier_c": classifier_c,
                "classification_threshold": threshold,
                **metrics,
                "maximum_predicted_class_share": maximum_share,
                "selection_scope": "outer_training_meta_inner_grouped_oof_only",
            }
            rank = (
                float(metrics["macro_f1"]),
                float(metrics["f1_label0"]),
                -abs(float(threshold) - 0.5),
                -abs(float(np.log10(classifier_c))),
                -float(threshold),
            )
            candidates.append((rank, float(classifier_c), float(threshold), record))
    if not candidates:
        raise RuntimeError("Every predeclared E4 configuration collapsed")
    _, classifier_c, threshold, record = max(candidates, key=lambda item: item[0])
    return classifier_c, threshold, record, probabilities_by_c[classifier_c]


def _base_summaries(directories: dict[str, Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, directory in directories.items():
        result[name] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))[
            "aggregate_repeated_oof"
        ]
    return result


def run_e4(
    train_path: Path,
    output_dir: Path,
    folds_path: Path,
    e0_dir: Path,
    e1_dir: Path,
    e2_dir: Path,
    e3_dir: Path,
) -> dict[str, Any]:
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"E4 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    frame = load_labeled_json(train_path, expected_sha256=OFFICIAL_SAMPLE_SHA256).reset_index(
        drop=True
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    presence = np.asarray([has_context(value) for value in frame["context"]], dtype=bool)
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    group_audit = build_v5_groups(frame)
    groups = np.asarray(group_audit.group_ids, dtype=object)
    assignments = build_common_fold_assignments(frame)
    if not assignments.equals(pd.read_csv(folds_path)):
        raise RuntimeError("Frozen common-fold assignment differs from regenerated folds")
    training = add_meta_features(_merge_training(e0_dir, e1_dir, e2_dir))
    validation = add_meta_features(_merge_oof(e0_dir, e1_dir, e2_dir))
    validation.to_csv(output_dir / "base_prediction_matrix.csv", index=False)
    training.to_csv(output_dir / "base_crossfit_training_matrix.csv", index=False)

    oof_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            fold_started = perf_counter()
            outer_train, outer_validation = outer_indices(
                assignments, row_ids, seed=seed, outer_fold=outer_fold
            )
            train_table = training.loc[
                (training["seed"] == seed)
                & (training["target_outer_fold"] == outer_fold)
            ].copy()
            validation_table = validation.loc[
                (validation["seed"] == seed) & (validation["outer_fold"] == outer_fold)
            ].copy()
            if set(train_table["row_id"].astype(str)) != set(row_ids[outer_train]):
                raise DataValidationError("E4 cross-fit training IDs are incomplete")
            if set(validation_table["row_id"].astype(str)) != set(row_ids[outer_validation]):
                raise DataValidationError("E4 outer base-prediction IDs are incomplete")
            classifier_c, threshold, inner_record, inner_meta_probabilities = select_inner_configuration(
                train_table,
                labels,
                row_ids,
                groups,
                outer_train,
                seed=seed,
                outer_fold=outer_fold,
            )
            train_features = train_table[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
            valid_features = validation_table[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
            model = _fit_model(
                train_features, train_table["label"].to_numpy(dtype=np.int64), classifier_c
            )
            probabilities = _predict_model(model, valid_features)
            model["metadata"] = {
                "classifier_c": classifier_c,
                "classification_threshold": threshold,
                "features": FEATURE_COLUMNS,
                "probability_orientation": "probability_label_1_is_faithful",
            }
            model_path = models_dir / f"seed_{seed}_fold_{outer_fold}.joblib"
            joblib.dump(model, model_path, compress=3)
            reloaded = joblib.load(model_path)
            reload_probabilities = _predict_model(reloaded, valid_features)
            reload_max_abs_diff = float(np.max(np.abs(reload_probabilities - probabilities)))
            if reload_max_abs_diff > 1e-12:
                raise RuntimeError("Reloaded E4 meta-model predictions differ")
            predictions = predictions_from_label1(probabilities, threshold)
            truth = validation_table["label"].to_numpy(dtype=np.int64)
            route = validation_table["context_present"].to_numpy(dtype=bool)
            fold_metrics.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "classifier_c": classifier_c,
                    "classification_threshold": threshold,
                    "inner_selection": inner_record,
                    "inner_meta_crossfit_rows": len(inner_meta_probabilities),
                    "group_overlap_count": 0,
                    "runtime_seconds": perf_counter() - fold_started,
                    "peak_vram_bytes": None,
                    "reload_max_abs_probability_difference": reload_max_abs_diff,
                    **metric_record(truth, probabilities, predictions, route),
                }
            )
            oof_records.append(
                pd.DataFrame(
                    {
                        "row_id": validation_table["row_id"].astype(str),
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "label": truth,
                        "context_present": route,
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
                    "reload_max_abs_probability_difference": reload_max_abs_diff,
                }
            )

    oof = pd.concat(oof_records, ignore_index=True).sort_values(
        ["seed", "outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_probability_frame(oof, row_ids)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
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
        oof["label"], oof["probability_label_1"], oof["prediction"], oof["context_present"]
    )
    pd.DataFrame(
        [{"route": name, **values} for name, values in aggregate["route_metrics"].items()]
    ).to_csv(output_dir / "route_metrics.csv", index=False)
    baselines = _base_summaries({"e0": e0_dir, "e1": e1_dir, "e2": e2_dir, "e3": e3_dir})
    seed_gains = {
        str(record["seed"]): float(record["macro_f1"] - next(
            item["macro_f1"]
            for item in json.loads((e0_dir / "summary.json").read_text(encoding="utf-8"))["seed_metrics"]
            if int(item["seed"]) == int(record["seed"])
        ))
        for record in seed_records
    }
    delta_e0 = float(aggregate["macro_f1"] - baselines["e0"]["macro_f1"])
    gate = {
        "delta_macro_f1_vs_e0": delta_e0,
        "maximum_predicted_class_share": aggregate["maximum_predicted_class_share"],
        "seeds_with_positive_gain_vs_e0": int(sum(value > 0 for value in seed_gains.values())),
        "seed_gains_vs_e0": seed_gains,
        "passes_primary_gate": bool(
            delta_e0 >= 0.01
            and aggregate["maximum_predicted_class_share"] <= 0.90
            and sum(value > 0 for value in seed_gains.values()) >= 2
        ),
    }
    summary = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "probability_orientation": "probability_label_1_is_faithful_label_1",
        "official_rows": len(frame),
        "oof_rows": len(oof),
        "complete_oof_predictions_per_official_row": len(OUTER_SEEDS),
        "aggregate_repeated_oof": aggregate,
        "seed_metrics": seed_records,
        "common_fold_comparators": {
            name: {
                "macro_f1": values["macro_f1"],
                "delta_e4_minus_comparator": float(aggregate["macro_f1"] - values["macro_f1"]),
            }
            for name, values in baselines.items()
        },
        "promotion_gate": gate,
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
        },
    )
    _write_json(
        output_dir / "config.json",
        {
            "experiment": EXPERIMENT,
            "base_models": ["e0", "e1", "e2"],
            "feature_columns": FEATURE_COLUMNS,
            "classifier_c_grid": C_GRID,
            "classification_threshold_grid": THRESHOLD_GRID,
            "meta_training_base_probability_scope": "outer_train_inner_grouped_cross_fit",
            "selection_scope": "outer_training_meta_inner_grouped_oof_only",
            "test_data_role": "not_accessed",
        },
    )
    _write_json(output_dir / "model_manifest.json", model_manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7/e4"))
    parser.add_argument("--folds-path", type=Path, default=Path("artifacts/v7/folds/common_folds.csv"))
    parser.add_argument("--e0-dir", type=Path, default=Path("artifacts/v7/e0"))
    parser.add_argument("--e1-dir", type=Path, default=Path("artifacts/v7/e1"))
    parser.add_argument("--e2-dir", type=Path, default=Path("artifacts/v7/e2"))
    parser.add_argument("--e3-dir", type=Path, default=Path("artifacts/v7/e3"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_e4(
        args.train_data,
        args.output_dir,
        args.folds_path,
        args.e0_dir,
        args.e1_dir,
        args.e2_dir,
        args.e3_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "experiment": summary["experiment"],
                "macro_f1": summary["aggregate_repeated_oof"]["macro_f1"],
                "promotion_gate": summary["promotion_gate"]["passes_primary_gate"],
                "test_rows_accessed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
