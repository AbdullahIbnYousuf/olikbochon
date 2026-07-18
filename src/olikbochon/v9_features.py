"""Frozen V9 retrieval/NLI features and grouped classifier evaluation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loading import validate_labeled_frame
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v4_preprocessing import official_context_is_present
from .v4_validation import VALIDATION_SEEDS
from .v5_features import CONTEXT_ABSENT_FEATURES, V5FeatureExtractor
from .v5_lexical import build_v5_folds, deterministic_substring_prediction
from .v7_features import TOP5_FEATURES, feature_vector, top5_features
from .v7_retrieval import RetrievedPassage
from .v9_nli import NLI_FEATURES


CANDIDATE_R = "candidate_r_mdeberta_semantic_verifier"
CANDIDATE_S = "candidate_s_xlmr_semantic_verifier"
CANDIDATE_T = "candidate_t_dual_semantic_verifier"
CANDIDATES = (CANDIDATE_R, CANDIDATE_S, CANDIDATE_T)
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169
NULL_CHAMPION = 0.541763
ROUTED_CHAMPION = 0.692051
CLASSIFIER_CONFIG = {
    "class_weight": "balanced",
    "C": 1.0,
    "max_iter": 3000,
    "solver": "liblinear",
    "random_state": 42,
}


def retrieval_feature_matrix(
    frame: pd.DataFrame,
    evidence: Sequence[Sequence[RetrievedPassage]],
) -> np.ndarray:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    if len(validated) != len(evidence) or any(len(row) != 5 for row in evidence):
        raise ValueError("V9 retrieval evidence is not aligned top-5 data")
    rows = [
        feature_vector(
            top5_features(row.response_bn, evidence[position], cutoff=0.0),
            TOP5_FEATURES,
        )
        for position, row in enumerate(validated.itertuples(index=False))
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(validated), len(TOP5_FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError("Candidate N retrieval feature matrix is invalid")
    return matrix


def frozen_feature_fingerprint() -> str:
    config = {
        "candidate_r": list(NLI_FEATURES) + list(TOP5_FEATURES),
        "candidate_s": list(NLI_FEATURES) + list(TOP5_FEATURES),
        "candidate_t": list(NLI_FEATURES) * 2
        + list(TOP5_FEATURES)
        + list(CONTEXT_ABSENT_FEATURES),
        "retrieval_cutoff": 0.0,
        "threshold": 0.5,
        "classifier": CLASSIFIER_CONFIG,
    }
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _index_hash(values: np.ndarray) -> str:
    array = np.sort(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(**CLASSIFIER_CONFIG)),
        ]
    )


def _prediction_counts(predictions: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def _metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {**classification_metrics(truth, predictions), "prediction_counts": _prediction_counts(predictions)}


def _label1_probability(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    logistic = model.named_steps["logistic"]
    column = list(logistic.classes_).index(1)
    values = model.predict_proba(matrix)[:, column].astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("V9 classifier probabilities contain NaN or Inf")
    return values


def _fold_task(payload: dict[str, Any]) -> dict[str, Any]:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    null_frame: pd.DataFrame = payload["null_frame"]
    labels = null_frame["label"].to_numpy(dtype=np.int64)
    train = np.asarray(payload["train_positions"], dtype=np.int64)
    validation = np.asarray(payload["validation_positions"], dtype=np.int64)
    if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[validation])) != {0, 1}:
        raise RuntimeError("V9 null-route fold lacks an official label")
    fixed = {
        CANDIDATE_R: np.hstack((payload["mdeberta"], payload["retrieval"])),
        CANDIDATE_S: np.hstack((payload["xlmr"], payload["retrieval"])),
    }
    extractor = V5FeatureExtractor.fit(null_frame.iloc[train].reset_index(drop=True))
    train_lexical = extractor.matrix(null_frame, train, context_present=False)
    validation_lexical = extractor.matrix(null_frame, validation, context_present=False)
    candidate_records: dict[str, Any] = {}
    returned_probabilities: dict[str, list[float]] = {}
    for candidate in CANDIDATES:
        if candidate == CANDIDATE_T:
            train_matrix = np.hstack(
                (
                    payload["mdeberta"][train],
                    payload["xlmr"][train],
                    payload["retrieval"][train],
                    train_lexical,
                )
            )
            validation_matrix = np.hstack(
                (
                    payload["mdeberta"][validation],
                    payload["xlmr"][validation],
                    payload["retrieval"][validation],
                    validation_lexical,
                )
            )
        else:
            train_matrix = fixed[candidate][train]
            validation_matrix = fixed[candidate][validation]
        if not np.isfinite(train_matrix).all() or not np.isfinite(validation_matrix).all():
            raise ValueError("V9 fold feature matrix contains NaN or Inf")
        model = _pipeline().fit(train_matrix, labels[train])
        probabilities = _label1_probability(model, validation_matrix)
        predictions = predictions_from_label1(probabilities, 0.5)
        candidate_records[candidate] = {
            **_metrics(labels[validation], predictions),
            "brier_score": float(np.mean(np.square(probabilities - labels[validation]))),
            "probability_mean": float(probabilities.mean()),
            "probability_std": float(probabilities.std(ddof=0)),
            "train_feature_rows": int(train_matrix.shape[0]),
            "validation_feature_rows": int(validation_matrix.shape[0]),
            "feature_count": int(train_matrix.shape[1]),
            "scaler_fit_rows": int(train_matrix.shape[0]),
            "classifier_fit_rows": int(train_matrix.shape[0]),
        }
        returned_probabilities[candidate] = probabilities.tolist()
    safe_record = {
        "seed": int(payload["seed"]),
        "fold": int(payload["fold"]),
        "train_rows": int(train.size),
        "validation_rows": int(validation.size),
        "train_groups": int(payload["train_groups"]),
        "validation_groups": int(payload["validation_groups"]),
        "group_overlap_count": 0,
        "training_index_sha256": str(payload["training_index_sha256"]),
        "validation_index_sha256": str(payload["validation_index_sha256"]),
        "feature_configuration_sha256": str(payload["feature_configuration_sha256"]),
        "validation_rows_used_for_fit": 0,
        "candidates": candidate_records,
        "raw_text_persisted": False,
        "row_level_probabilities_persisted": False,
    }
    fold_dir = Path(payload["fold_directory"])
    fold_dir.mkdir(parents=True, exist_ok=False)
    (fold_dir / "fold_result.json").write_text(
        json.dumps(safe_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **safe_record,
        "validation_positions": validation.tolist(),
        "probabilities": returned_probabilities,
    }


def _distribution(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([record["macro_f1"] for record in records], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _verdict(null_f1: float, routed_f1: float, seed_std: float, class_share: float) -> dict[str, Any]:
    guards = seed_std <= 0.04 and class_share <= 0.90
    null_level = (
        "high_value"
        if null_f1 >= 0.68
        else "strong"
        if null_f1 >= 0.62
        else "promising"
        if null_f1 >= 0.58
        else "rejected"
    )
    routed_level = (
        "high_value"
        if routed_f1 >= 0.74
        else "strong"
        if routed_f1 >= 0.72
        else "meaningful"
        if routed_f1 >= 0.70
        else "valid_improvement"
        if routed_f1 > ROUTED_CHAMPION
        else "rejected"
    )
    accepted = guards and null_f1 > NULL_CHAMPION and routed_f1 > ROUTED_CHAMPION
    return {
        "null_gate": null_level,
        "routed_gate": routed_level,
        "seed_stability_guard": seed_std <= 0.04,
        "class_collapse_guard": class_share <= 0.90,
        "accepted": accepted,
    }


def evaluate_candidates(
    frame: pd.DataFrame,
    mdeberta: np.ndarray,
    xlmr: np.ndarray,
    retrieval: np.ndarray,
    *,
    fold_workers: int,
    temporary_root: Path,
) -> dict[str, Any]:
    """Run exactly 15 grouped outer evaluations, up to three spawn workers."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V9 corrected route totals differ from 130/169")
    absent_indices = np.flatnonzero(~presence)
    null_frame = validated.iloc[absent_indices].reset_index(drop=True)
    expected_shape = (EXPECTED_ABSENT_ROWS,)
    if any(matrix.shape[0] != expected_shape[0] for matrix in (mdeberta, xlmr, retrieval)):
        raise ValueError("V9 frozen feature rows are not aligned")
    folds = build_v5_folds(validated)
    group_ids = np.asarray(folds.audit.group_ids, dtype=object)
    positions = {int(index): position for position, index in enumerate(absent_indices)}
    fingerprint = frozen_feature_fingerprint()
    payloads: list[dict[str, Any]] = []
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        if set(group_ids[train]) & set(group_ids[validation]):
            raise RuntimeError("V9 outer grouped split leaked a strengthened group")
        train_absent = train[~presence[train]]
        validation_absent = validation[~presence[validation]]
        train_positions = np.asarray([positions[int(index)] for index in train_absent])
        validation_positions = np.asarray([positions[int(index)] for index in validation_absent])
        payloads.append(
            {
                "seed": split.seed,
                "fold": split.fold,
                "null_frame": null_frame,
                "mdeberta": mdeberta,
                "xlmr": xlmr,
                "retrieval": retrieval,
                "train_positions": train_positions,
                "validation_positions": validation_positions,
                "train_groups": len(set(group_ids[train_absent])),
                "validation_groups": len(set(group_ids[validation_absent])),
                "training_index_sha256": _index_hash(train_absent),
                "validation_index_sha256": _index_hash(validation_absent),
                "feature_configuration_sha256": fingerprint,
                "fold_directory": str(temporary_root / f"seed{split.seed}_fold{split.fold}"),
            }
        )
    context = multiprocessing.get_context("spawn")
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=fold_workers, mp_context=context) as executor:
        futures = [executor.submit(_fold_task, payload) for payload in payloads]
        for future in as_completed(futures):
            completed.append(future.result())
    expected = {(seed, fold) for seed in VALIDATION_SEEDS for fold in range(1, 6)}
    observed = {(record["seed"], record["fold"]) for record in completed}
    if observed != expected or len(completed) != len(expected):
        raise RuntimeError("V9 grouped workers did not complete every seed/fold exactly once")
    completed.sort(key=lambda record: (record["seed"], record["fold"]))
    labels = validated["label"].to_numpy(dtype=np.int64)
    null_truth = labels[~presence]
    present_predictions = np.asarray(
        [
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in validated.loc[presence].itertuples(index=False)
        ],
        dtype=np.int64,
    )
    candidates: dict[str, Any] = {}
    for candidate in CANDIDATES:
        seed_null_records: list[dict[str, Any]] = []
        seed_routed_records: list[dict[str, Any]] = []
        pooled_probabilities: list[np.ndarray] = []
        pooled_null_predictions: list[np.ndarray] = []
        pooled_routed_predictions: list[np.ndarray] = []
        for seed in VALIDATION_SEEDS:
            probabilities = np.full(EXPECTED_ABSENT_ROWS, np.nan, dtype=np.float64)
            coverage = np.zeros(EXPECTED_ABSENT_ROWS, dtype=np.int64)
            for record in (item for item in completed if item["seed"] == seed):
                validation = np.asarray(record["validation_positions"], dtype=np.int64)
                probabilities[validation] = np.asarray(record["probabilities"][candidate])
                coverage[validation] += 1
            if not np.all(coverage == 1) or not np.isfinite(probabilities).all():
                raise RuntimeError("V9 OOF coverage is incomplete")
            null_predictions = predictions_from_label1(probabilities, 0.5)
            routed = np.empty(len(validated), dtype=np.int64)
            routed[presence] = present_predictions
            routed[~presence] = null_predictions
            seed_null_records.append(_metrics(null_truth, null_predictions))
            seed_routed_records.append(_metrics(labels, routed))
            pooled_probabilities.append(probabilities)
            pooled_null_predictions.append(null_predictions)
            pooled_routed_predictions.append(routed)
        repeated_null_truth = np.tile(null_truth, len(VALIDATION_SEEDS))
        repeated_truth = np.tile(labels, len(VALIDATION_SEEDS))
        repeated_presence = np.tile(presence, len(VALIDATION_SEEDS))
        probabilities = np.concatenate(pooled_probabilities)
        null_predictions = np.concatenate(pooled_null_predictions)
        routed_predictions = np.concatenate(pooled_routed_predictions)
        null_metrics = _metrics(repeated_null_truth, null_predictions)
        routed_metrics = {
            **_metrics(repeated_truth, routed_predictions),
            "route_metrics": subgroup_metrics(
                repeated_truth, routed_predictions, repeated_presence
            ),
        }
        null_distribution = _distribution(seed_null_records)
        routed_distribution = _distribution(seed_routed_records)
        class_share = float(np.bincount(null_predictions, minlength=2).max() / len(null_predictions))
        candidates[candidate] = {
            "null_route": {
                **null_metrics,
                "threshold": 0.5,
                "brier_score": float(np.mean(np.square(probabilities - repeated_null_truth))),
                "probability_mean": float(probabilities.mean()),
                "probability_std": float(probabilities.std(ddof=0)),
                "seed_macro_f1_distribution": null_distribution,
                "maximum_predicted_class_share": class_share,
                "passes_class_collapse_guard": class_share <= 0.90,
            },
            "routed": {
                **routed_metrics,
                "threshold": 0.5,
                "seed_macro_f1_distribution": routed_distribution,
            },
            "verdict": _verdict(
                float(null_metrics["macro_f1"]),
                float(routed_metrics["macro_f1"]),
                float(null_distribution["std"]),
                class_share,
            ),
        }
    safe_folds = [
        {name: value for name, value in record.items() if name not in {"probabilities", "validation_positions"}}
        for record in completed
    ]
    return {
        "candidates": candidates,
        "fold_metrics": safe_folds,
        "validation": {
            "seeds": list(VALIDATION_SEEDS),
            "folds_per_seed": 5,
            "outer_evaluations": 15,
            "complete_oof_coverage_each_seed": True,
            "zero_group_overlap": True,
            "both_labels_in_every_null_split": True,
            "feature_configuration_sha256": fingerprint,
            "fold_workers": fold_workers,
            "multiprocessing_start_method": "spawn",
            "blas_threads_per_worker": 1,
        },
        "raw_text_persisted": False,
        "row_level_probabilities_persisted": False,
    }
