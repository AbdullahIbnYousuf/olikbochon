"""Common-fold E0 regeneration of the verified V4-A retrieval/lexical system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .metrics import classification_metrics, predictions_from_label1
from .v4_wikipedia import (
    RETRIEVAL_THRESHOLDS,
    V4Files,
    WikipediaRetriever,
    RetrievalResult,
    _wiki_chunks,
    build_lexical_frame,
    load_wikipedia_corpus,
)
from .v5_lexical import build_v5_groups
from .v7_e1 import _write_json
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


EXPERIMENT = "e0_common_fold_v4a"
CLASSIFICATION_THRESHOLDS = tuple(round(value / 100, 2) for value in range(40, 61, 2))


def effective_contexts(
    frame: pd.DataFrame, retrieval: RetrievalResult, retrieval_threshold: float
) -> list[str]:
    if len(frame) != len(retrieval.snippets) or len(frame) != len(retrieval.similarities):
        raise ValueError("Retrieval output must align with the labeled frame")
    result: list[str] = []
    for index, value in enumerate(frame["context"]):
        if has_context(value):
            result.append(str(value).strip())
        elif float(retrieval.similarities[index]) >= retrieval_threshold:
            result.append(str(retrieval.snippets[index]))
        else:
            result.append("")
    return result


def _fit_predict(
    features: pd.DataFrame,
    labels: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(features.iloc[train])
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ).fit(scaler.transform(features.iloc[train]), labels[train])
    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(scaler.transform(features.iloc[validation]))[
        :, class_index
    ].astype(np.float64)
    return probabilities, {"scaler": scaler, "classifier": model}


def select_inner_configuration(
    frame: pd.DataFrame,
    labels: np.ndarray,
    row_ids: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    features_by_retrieval_threshold: dict[float, pd.DataFrame],
    *,
    seed: int,
    outer_fold: int,
) -> tuple[float, float, dict[str, Any], np.ndarray]:
    inner_folds = make_inner_grouped_folds(
        labels, groups, outer_train, seed=seed, outer_fold=outer_fold
    )
    ranked: list[tuple[tuple[float, ...], float, float, dict[str, Any]]] = []
    probabilities_by_threshold: dict[float, np.ndarray] = {}
    for retrieval_threshold, features in features_by_retrieval_threshold.items():
        probabilities = np.full(len(frame), np.nan, dtype=np.float64)
        for split in inner_folds:
            train = np.asarray(split.train_indices, dtype=np.int64)
            validation = np.asarray(split.validation_indices, dtype=np.int64)
            FitScope(
                "e0_inner_scaler_classifier_fit",
                tuple(row_ids[train]),
                tuple(row_ids[validation]),
            ).validate()
            SelectionAudit(
                "retrieval_acceptance_threshold",
                tuple(row_ids[train]),
                tuple(row_ids[validation]),
            ).validate()
            SelectionAudit(
                "classification_threshold",
                tuple(row_ids[train]),
                tuple(row_ids[validation]),
            ).validate()
            probabilities[validation], _ = _fit_predict(features, labels, train, validation)
        selected_probabilities = probabilities[outer_train]
        if not np.isfinite(selected_probabilities).all():
            raise RuntimeError("E0 inner cross-fit did not cover outer-training rows")
        probabilities_by_threshold[retrieval_threshold] = selected_probabilities.copy()
        for classification_threshold in CLASSIFICATION_THRESHOLDS:
            predictions = predictions_from_label1(
                selected_probabilities, classification_threshold
            )
            metrics = classification_metrics(labels[outer_train], predictions)
            counts = np.bincount(predictions, minlength=2)
            share = float(counts.max() / counts.sum())
            if share > 0.90:
                continue
            record = {
                "retrieval_threshold": retrieval_threshold,
                "classification_threshold": classification_threshold,
                **metrics,
                "maximum_predicted_class_share": share,
                "selection_scope": "outer_training_inner_grouped_oof_only",
            }
            rank = (
                metrics["macro_f1"],
                metrics["f1_label0"],
                -abs(classification_threshold - 0.5),
                -abs(retrieval_threshold - 0.25),
                -classification_threshold,
                -retrieval_threshold,
            )
            ranked.append(
                (rank, retrieval_threshold, classification_threshold, record)
            )
    if not ranked:
        raise RuntimeError("Every predeclared E0 inner configuration collapsed")
    _, retrieval_threshold, classification_threshold, record = max(
        ranked, key=lambda item: item[0]
    )
    return (
        float(retrieval_threshold),
        float(classification_threshold),
        record,
        probabilities_by_threshold[float(retrieval_threshold)],
    )


def run_e0(
    frame: pd.DataFrame,
    retrieval: RetrievalResult,
    output_dir: Path,
    folds_path: Path,
    *,
    resource_manifest: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    labels = frame["label"].to_numpy(dtype=np.int64)
    presence = np.asarray([has_context(value) for value in frame["context"]], dtype=bool)
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    group_audit = build_v5_groups(frame)
    groups = np.asarray(group_audit.group_ids, dtype=object)
    assignments = build_common_fold_assignments(frame)
    assignments.to_csv(folds_path, index=False)
    features = {
        threshold: build_lexical_frame(
            effective_contexts(frame, retrieval, threshold), frame["response_bn"]
        )
        for threshold in RETRIEVAL_THRESHOLDS
    }
    oof_records: list[pd.DataFrame] = []
    stack_records: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            fold_started = perf_counter()
            train, validation = outer_indices(
                assignments, row_ids, seed=seed, outer_fold=outer_fold
            )
            FitScope("e0_outer_fit", tuple(row_ids[train]), tuple(row_ids[validation])).validate()
            retrieval_threshold, threshold, inner_record, inner_probabilities = (
                select_inner_configuration(
                    frame,
                    labels,
                    row_ids,
                    groups,
                    train,
                    features,
                    seed=seed,
                    outer_fold=outer_fold,
                )
            )
            probabilities, state = _fit_predict(
                features[retrieval_threshold], labels, train, validation
            )
            predictions = predictions_from_label1(probabilities, threshold)
            model_path = models_dir / f"seed_{seed}_fold_{outer_fold}.joblib"
            joblib.dump(
                {
                    **state,
                    "retrieval_threshold": retrieval_threshold,
                    "classification_threshold": threshold,
                },
                model_path,
                compress=3,
            )
            reloaded = joblib.load(model_path)
            class_index = list(reloaded["classifier"].classes_).index(1)
            reloaded_probabilities = reloaded["classifier"].predict_proba(
                reloaded["scaler"].transform(features[retrieval_threshold].iloc[validation])
            )[:, class_index]
            reload_difference = float(np.max(np.abs(reloaded_probabilities - probabilities)))
            if reload_difference > 1e-12:
                raise RuntimeError("Reloaded E0 predictions differ")
            selected_contexts = effective_contexts(frame, retrieval, retrieval_threshold)
            accepted = (~presence[validation]) & (
                retrieval.similarities[validation] >= retrieval_threshold
            )
            fold_records.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "retrieval_threshold": retrieval_threshold,
                    "classification_threshold": threshold,
                    "inner_selection": inner_record,
                    "group_overlap_count": 0,
                    "runtime_seconds": perf_counter() - fold_started,
                    "peak_vram_bytes": None,
                    "reload_max_abs_probability_difference": reload_difference,
                    **metric_record(
                        labels[validation], probabilities, predictions, presence[validation]
                    ),
                }
            )
            oof_records.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[validation],
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "label": labels[validation],
                        "context_present": presence[validation],
                        "probability_label_0": 1.0 - probabilities,
                        "probability_label_1": probabilities,
                        "classification_threshold": threshold,
                        "prediction": predictions,
                        "retrieval_score": retrieval.similarities[validation],
                        "retrieval_accepted": accepted,
                        "evidence_length": [len(selected_contexts[index]) for index in validation],
                    }
                )
            )
            stack_records.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[train],
                        "seed": seed,
                        "target_outer_fold": outer_fold,
                        "label": labels[train],
                        "context_present": presence[train],
                        "probability_label_1_e0": inner_probabilities,
                        "probability_label_0_e0": 1.0 - inner_probabilities,
                        "base_prediction_provenance": "outer_train_inner_grouped_cross_fit",
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
                    "reload_max_abs_probability_difference": reload_difference,
                }
            )
    oof = pd.concat(oof_records, ignore_index=True).sort_values(
        ["seed", "outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_probability_frame(oof, row_ids)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.concat(stack_records, ignore_index=True).sort_values(
        ["seed", "target_outer_fold", "row_id"], kind="mergesort"
    ).to_csv(output_dir / "stack_training_predictions.csv", index=False)
    pd.DataFrame(fold_records).drop(columns=["route_metrics", "inner_selection"]).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    seed_records = []
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
    summary = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "aggregate_repeated_oof": aggregate,
        "seed_metrics": seed_records,
        "group_audit": group_audit.__dict__,
        "calibration": calibration_table(oof["label"], oof["probability_label_1"]),
        "runtime_seconds": perf_counter() - started,
        "peak_vram_bytes": None,
        "test_rows_accessed": 0,
        "resource_manifest": resource_manifest,
        "reload_all_equal": True,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "config.json",
        {
            "experiment": EXPERIMENT,
            "seeds": OUTER_SEEDS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": 3,
            "retrieval_thresholds": RETRIEVAL_THRESHOLDS,
            "classification_thresholds": CLASSIFICATION_THRESHOLDS,
            "feature_family": "verified_v4a_eight_lexical_numeric_features",
            "selection_scope": "nested_inner_grouped_only",
        },
    )
    _write_json(output_dir / "confusion_matrices.json", {"aggregate": aggregate})
    _write_json(output_dir / "model_manifest.json", model_manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--wikipedia-root", type=Path, required=True)
    parser.add_argument("--resource-name", default="abyaadrafid/bnwiki")
    parser.add_argument("--resource-version", default="1")
    parser.add_argument("--resource-license", default="CC0-1.0")
    parser.add_argument("--resource-source-url", default=None)
    parser.add_argument("--resource-sha1", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7/e0"))
    parser.add_argument(
        "--folds-path", type=Path, default=Path("artifacts/v7/folds/common_folds.csv")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = load_labeled_json(args.train_data, expected_sha256=OFFICIAL_SAMPLE_SHA256)
    chunks = _wiki_chunks(args.wikipedia_root)
    if not chunks:
        raise FileNotFoundError(
            "A verified AA/AB/AC/AD Bengali Wikipedia WikiExtractor corpus is required"
        )
    placeholder = args.train_data
    files = V4Files(
        args.train_data,
        placeholder,
        placeholder,
        args.train_data.parent,
        args.wikipedia_root,
        chunks,
        0,
    )
    corpus = load_wikipedia_corpus(files)
    retrieval = WikipediaRetriever(corpus).retrieve(frame["prompt_bn"])
    summary = run_e0(
        frame,
        retrieval,
        args.output_dir,
        args.folds_path,
        resource_manifest={
            "wikipedia_root": str(args.wikipedia_root.resolve()),
            "chunk_count": len(chunks),
            "aggregate_bytes": sum(path.stat().st_size for path in chunks),
            "dataset_ref": args.resource_name,
            "dataset_version": args.resource_version,
            "license_metadata": args.resource_license,
            "source_url": args.resource_source_url,
            "source_dump_sha1": args.resource_sha1,
        },
    )
    print(json.dumps({"status": summary["status"], "experiment": EXPERIMENT}))


if __name__ == "__main__":
    main()
