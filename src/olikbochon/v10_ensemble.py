"""Genuinely nested V10 lexical-semantic probability ensembles."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import Counter
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
from .v5_lexical import build_v5_folds, deterministic_substring_prediction
from .v5_sparse import CANDIDATES as V5_SPARSE_CANDIDATES
from .v5_sparse import SparseNullModel
from .v7_runner import build_inner_splits
from .v9_features import CLASSIFIER_CONFIG
from .v9_nli import NLI_FEATURES


CANDIDATE_I = V5_SPARSE_CANDIDATES[4]
CANDIDATE_R = "candidate_r_mdeberta_semantic_verifier"
CANDIDATE_U = "candidate_u_fixed_equal_average"
CANDIDATE_V = "candidate_v_nested_convex_blend"
CANDIDATE_W = "candidate_w_nested_logistic_stacker"
CANDIDATES = (CANDIDATE_U, CANDIDATE_V, CANDIDATE_W)
BLEND_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)
DECISION_THRESHOLD = 0.50
ELIGIBLE_ROUTED_F1 = 0.697051
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(**CLASSIFIER_CONFIG)),
        ]
    )


def _label1_probability(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    logistic = model.named_steps["logistic"]
    values = model.predict_proba(matrix)[:, list(logistic.classes_).index(1)]
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("V10 probabilities contain NaN or Inf")
    return values


def _index_hash(indices: np.ndarray) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def feature_configuration_fingerprint() -> str:
    config = {
        "candidate_i": CANDIDATE_I,
        "candidate_r": list(NLI_FEATURES) + ["candidate_n_retrieval_features_7"],
        "candidate_u_candidate_i_weight": 0.5,
        "candidate_v_candidate_i_weights": list(BLEND_WEIGHTS),
        "candidate_w_inputs": ["candidate_i_probability", "candidate_r_probability"],
        "classifier": CLASSIFIER_CONFIG,
        "decision_threshold": DECISION_THRESHOLD,
        "inner_folds": 3,
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=2)
    return {
        **classification_metrics(truth, predictions),
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
    }


def select_nested_weight(
    truth: np.ndarray,
    candidate_i_probability: np.ndarray,
    candidate_r_probability: np.ndarray,
) -> dict[str, Any]:
    """Select only the frozen Candidate-I weight grid using inner OOF predictions."""
    records: list[dict[str, Any]] = []
    for weight in BLEND_WEIGHTS:
        probability = weight * candidate_i_probability + (1.0 - weight) * candidate_r_probability
        metrics = _metrics(truth, predictions_from_label1(probability, DECISION_THRESHOLD))
        records.append({"candidate_i_weight": weight, **metrics})
    selected = max(
        records,
        key=lambda record: (
            record["macro_f1"],
            record["f1_label0"],
            -abs(record["candidate_i_weight"] - 0.50),
            -record["candidate_i_weight"],
        ),
    )
    return {
        "candidate_i_weight": float(selected["candidate_i_weight"]),
        "inner_metrics": selected,
        "grid_metrics": records,
        "selection_scope": "current_outer_training_partition_inner_grouped_oof_only",
        "outer_validation_rows_used": 0,
    }


def _base_probabilities(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    r_train: np.ndarray,
    r_validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_i = SparseNullModel(CANDIDATE_I).fit(train_frame)
    probability_i = candidate_i.predict_label1_probability(validation_frame)
    candidate_r = _pipeline().fit(r_train, train_frame["label"].to_numpy(dtype=np.int64))
    probability_r = _label1_probability(candidate_r, r_validation)
    return probability_i, probability_r, candidate_i.fit_audit()


def _fold_task(payload: dict[str, Any]) -> dict[str, Any]:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    frame: pd.DataFrame = payload["null_frame"]
    r_matrix = np.asarray(payload["candidate_r_matrix"], dtype=np.float64)
    labels = frame["label"].to_numpy(dtype=np.int64)
    train = np.asarray(payload["train_positions"], dtype=np.int64)
    validation = np.asarray(payload["validation_positions"], dtype=np.int64)
    groups = np.asarray(payload["null_group_ids"], dtype=object)
    inner_splits = build_inner_splits(
        labels[train],
        groups[train],
        outer_seed=int(payload["seed"]),
        outer_fold=int(payload["fold"]),
    )
    inner_i = np.full(train.size, np.nan, dtype=np.float64)
    inner_r = np.full(train.size, np.nan, dtype=np.float64)
    inner_coverage = np.zeros(train.size, dtype=np.int64)
    inner_provenance: list[dict[str, Any]] = []
    for inner_fold, (inner_train_local, inner_validation_local) in enumerate(
        inner_splits, start=1
    ):
        inner_train = train[inner_train_local]
        inner_validation = train[inner_validation_local]
        if set(groups[inner_train]) & set(groups[inner_validation]):
            raise RuntimeError("V10 inner grouped split leaked a strengthened group")
        probability_i, probability_r, audit = _base_probabilities(
            frame.iloc[inner_train].reset_index(drop=True),
            frame.iloc[inner_validation].reset_index(drop=True),
            r_matrix[inner_train],
            r_matrix[inner_validation],
        )
        inner_i[inner_validation_local] = probability_i
        inner_r[inner_validation_local] = probability_r
        inner_coverage[inner_validation_local] += 1
        inner_provenance.append(
            {
                "inner_fold": inner_fold,
                "training_index_sha256": _index_hash(
                    np.asarray(payload["official_absent_indices"])[inner_train]
                ),
                "validation_index_sha256": _index_hash(
                    np.asarray(payload["official_absent_indices"])[inner_validation]
                ),
                "train_rows": int(inner_train.size),
                "validation_rows": int(inner_validation.size),
                "group_overlap_count": 0,
                "candidate_i_training_rows": audit["training_row_count"],
                "outer_validation_rows_used": 0,
            }
        )
    if (
        not np.all(inner_coverage == 1)
        or not np.isfinite(inner_i).all()
        or not np.isfinite(inner_r).all()
    ):
        raise RuntimeError("V10 inner OOF base probabilities are incomplete")
    nested_weight = select_nested_weight(labels[train], inner_i, inner_r)
    stacker = _pipeline().fit(np.column_stack((inner_i, inner_r)), labels[train])
    outer_i, outer_r, outer_i_audit = _base_probabilities(
        frame.iloc[train].reset_index(drop=True),
        frame.iloc[validation].reset_index(drop=True),
        r_matrix[train],
        r_matrix[validation],
    )
    probability_u = 0.5 * outer_i + 0.5 * outer_r
    weight = float(nested_weight["candidate_i_weight"])
    probability_v = weight * outer_i + (1.0 - weight) * outer_r
    probability_w = _label1_probability(stacker, np.column_stack((outer_i, outer_r)))
    probability_blocks = {
        CANDIDATE_I: outer_i,
        CANDIDATE_R: outer_r,
        CANDIDATE_U: probability_u,
        CANDIDATE_V: probability_v,
        CANDIDATE_W: probability_w,
    }
    safe_candidates = {
        candidate: {
            **_metrics(
                labels[validation],
                predictions_from_label1(probability, DECISION_THRESHOLD),
            ),
            "brier_score": float(np.mean(np.square(probability - labels[validation]))),
            "probability_mean": float(probability.mean()),
            "probability_std": float(probability.std(ddof=0)),
        }
        for candidate, probability in probability_blocks.items()
    }
    logistic = stacker.named_steps["logistic"]
    record = {
        "seed": int(payload["seed"]),
        "fold": int(payload["fold"]),
        "train_rows": int(train.size),
        "validation_rows": int(validation.size),
        "train_groups": int(len(set(groups[train]))),
        "validation_groups": int(len(set(groups[validation]))),
        "group_overlap_count": 0,
        "training_index_sha256": str(payload["training_index_sha256"]),
        "validation_index_sha256": str(payload["validation_index_sha256"]),
        "feature_configuration_sha256": str(payload["feature_configuration_sha256"]),
        "inner_oof": {
            "fold_count": len(inner_splits),
            "complete_coverage": True,
            "outer_validation_rows_used": 0,
            "folds": inner_provenance,
        },
        "candidate_v_selection": nested_weight,
        "candidate_w": {
            "inputs": ["candidate_i_probability", "candidate_r_probability"],
            "fit_rows": int(train.size),
            "outer_validation_rows_used": 0,
            "standardized_coefficients": {
                "candidate_i_probability": float(logistic.coef_[0, 0]),
                "candidate_r_probability": float(logistic.coef_[0, 1]),
            },
            "intercept": float(logistic.intercept_[0]),
        },
        "candidate_i_fit_audit": outer_i_audit,
        "candidates": safe_candidates,
        "raw_text_persisted": False,
        "row_level_probabilities_persisted": False,
    }
    fold_directory = Path(payload["fold_directory"])
    fold_directory.mkdir(parents=True, exist_ok=False)
    (fold_directory / "fold_result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        **record,
        "validation_positions": validation.tolist(),
        "probabilities": {
            candidate: probability.tolist()
            for candidate, probability in probability_blocks.items()
        },
    }


def _distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("macro_f1", "f1_label0", "f1_label1", "accuracy"):
        values = np.asarray([record[field] for record in records], dtype=np.float64)
        result[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return result


def _complementarity(
    truth: np.ndarray, probability_i: np.ndarray, probability_r: np.ndarray
) -> dict[str, Any]:
    prediction_i = predictions_from_label1(probability_i, DECISION_THRESHOLD)
    prediction_r = predictions_from_label1(probability_r, DECISION_THRESHOLD)
    correct_i = prediction_i == truth
    correct_r = prediction_r == truth
    return {
        "probability_pearson_correlation": float(np.corrcoef(probability_i, probability_r)[0, 1]),
        "prediction_disagreement_rate": float(np.mean(prediction_i != prediction_r)),
        "both_correct": int(np.sum(correct_i & correct_r)),
        "candidate_i_only_correct": int(np.sum(correct_i & ~correct_r)),
        "candidate_r_only_correct": int(np.sum(~correct_i & correct_r)),
        "both_wrong": int(np.sum(~correct_i & ~correct_r)),
    }


def run_nested_ensemble(
    frame: pd.DataFrame,
    candidate_r_matrix: np.ndarray,
    *,
    fold_workers: int,
    temporary_root: Path,
) -> dict[str, Any]:
    """Evaluate U/V/W with genuine inner OOF construction in every outer fold."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V10 corrected route totals differ from 130/169")
    absent_indices = np.flatnonzero(~presence)
    null_frame = validated.iloc[absent_indices].reset_index(drop=True)
    if candidate_r_matrix.shape[0] != EXPECTED_ABSENT_ROWS:
        raise ValueError("V10 Candidate R features are not aligned to the null route")
    folds = build_v5_folds(validated)
    full_groups = np.asarray(folds.audit.group_ids, dtype=object)
    null_groups = full_groups[absent_indices]
    null_positions = {int(index): position for position, index in enumerate(absent_indices)}
    fingerprint = feature_configuration_fingerprint()
    payloads: list[dict[str, Any]] = []
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        if set(full_groups[train]) & set(full_groups[validation]):
            raise RuntimeError("V10 outer grouped split leaked a strengthened group")
        train_absent = train[~presence[train]]
        validation_absent = validation[~presence[validation]]
        train_positions = np.asarray([null_positions[int(index)] for index in train_absent])
        validation_positions = np.asarray(
            [null_positions[int(index)] for index in validation_absent]
        )
        payloads.append(
            {
                "seed": split.seed,
                "fold": split.fold,
                "null_frame": null_frame,
                "candidate_r_matrix": candidate_r_matrix,
                "null_group_ids": null_groups,
                "official_absent_indices": absent_indices,
                "train_positions": train_positions,
                "validation_positions": validation_positions,
                "training_index_sha256": _index_hash(train_absent),
                "validation_index_sha256": _index_hash(validation_absent),
                "feature_configuration_sha256": fingerprint,
                "fold_directory": str(
                    temporary_root / f"seed{split.seed}_fold{split.fold}"
                ),
            }
        )
    context = multiprocessing.get_context("spawn")
    completed: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=fold_workers, mp_context=context) as executor:
        futures = [executor.submit(_fold_task, payload) for payload in payloads]
        for future in as_completed(futures):
            completed.append(future.result())
    expected = {(seed, fold) for seed in VALIDATION_SEEDS for fold in range(1, 6)}
    if {(item["seed"], item["fold"]) for item in completed} != expected or len(completed) != 15:
        raise RuntimeError("V10 outer workers did not finish every seed/fold exactly once")
    completed.sort(key=lambda record: (record["seed"], record["fold"]))
    truth = validated["label"].to_numpy(dtype=np.int64)
    null_truth = truth[~presence]
    present_predictions = np.asarray(
        [
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in validated.loc[presence].itertuples(index=False)
        ],
        dtype=np.int64,
    )
    all_names = (CANDIDATE_I, CANDIDATE_R) + CANDIDATES
    seed_results: dict[str, Any] = {}
    pooled: dict[str, list[np.ndarray]] = {candidate: [] for candidate in all_names}
    complementarity: list[dict[str, Any]] = []
    for seed in VALIDATION_SEEDS:
        seed_probabilities = {
            candidate: np.full(EXPECTED_ABSENT_ROWS, np.nan, dtype=np.float64)
            for candidate in all_names
        }
        coverage = np.zeros(EXPECTED_ABSENT_ROWS, dtype=np.int64)
        for record in (item for item in completed if item["seed"] == seed):
            validation = np.asarray(record["validation_positions"], dtype=np.int64)
            coverage[validation] += 1
            for candidate in all_names:
                seed_probabilities[candidate][validation] = np.asarray(
                    record["probabilities"][candidate], dtype=np.float64
                )
        if not np.all(coverage == 1) or any(
            not np.isfinite(values).all() for values in seed_probabilities.values()
        ):
            raise RuntimeError("V10 outer OOF coverage is incomplete")
        complementarity.append(
            _complementarity(
                null_truth,
                seed_probabilities[CANDIDATE_I],
                seed_probabilities[CANDIDATE_R],
            )
        )
        seed_record: dict[str, Any] = {}
        for candidate in all_names:
            probability = seed_probabilities[candidate]
            null_prediction = predictions_from_label1(probability, DECISION_THRESHOLD)
            routed = np.empty(len(validated), dtype=np.int64)
            routed[presence] = present_predictions
            routed[~presence] = null_prediction
            seed_record[candidate] = {
                "null": _metrics(null_truth, null_prediction),
                "routed": _metrics(truth, routed),
            }
            pooled[candidate].append(probability)
        seed_results[str(seed)] = seed_record
    repeated_null_truth = np.tile(null_truth, len(VALIDATION_SEEDS))
    repeated_truth = np.tile(truth, len(VALIDATION_SEEDS))
    repeated_presence = np.tile(presence, len(VALIDATION_SEEDS))
    aggregates: dict[str, Any] = {}
    for candidate in all_names:
        probabilities = np.concatenate(pooled[candidate])
        null_predictions = predictions_from_label1(probabilities, DECISION_THRESHOLD)
        routed_blocks: list[np.ndarray] = []
        null_seed_metrics: list[dict[str, Any]] = []
        routed_seed_metrics: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(VALIDATION_SEEDS):
            null_prediction = predictions_from_label1(
                pooled[candidate][seed_index], DECISION_THRESHOLD
            )
            routed = np.empty(len(validated), dtype=np.int64)
            routed[presence] = present_predictions
            routed[~presence] = null_prediction
            routed_blocks.append(routed)
            null_seed_metrics.append(seed_results[str(seed)][candidate]["null"])
            routed_seed_metrics.append(seed_results[str(seed)][candidate]["routed"])
        routed_predictions = np.concatenate(routed_blocks)
        aggregates[candidate] = {
            "null": {
                **_metrics(repeated_null_truth, null_predictions),
                "brier_score": float(
                    np.mean(np.square(probabilities - repeated_null_truth))
                ),
                "probability_mean": float(probabilities.mean()),
                "probability_std": float(probabilities.std(ddof=0)),
                "seed_distribution": _distribution(null_seed_metrics),
            },
            "routed": {
                **_metrics(repeated_truth, routed_predictions),
                "route_metrics": subgroup_metrics(
                    repeated_truth, routed_predictions, repeated_presence
                ),
                "seed_distribution": _distribution(routed_seed_metrics),
            },
        }
    eligible: list[str] = []
    for candidate in CANDIDATES:
        routed = aggregates[candidate]["routed"]
        null_predictions = predictions_from_label1(
            np.concatenate(pooled[candidate]), DECISION_THRESHOLD
        )
        class_share = float(
            np.bincount(null_predictions, minlength=2).max() / len(null_predictions)
        )
        paired_improvements = {
            str(seed): float(
                seed_results[str(seed)][candidate]["routed"]["macro_f1"]
                - seed_results[str(seed)][CANDIDATE_I]["routed"]["macro_f1"]
            )
            for seed in VALIDATION_SEEDS
        }
        improved_seed_count = sum(value > 0 for value in paired_improvements.values())
        stable = routed["seed_distribution"]["macro_f1"]["std"] <= 0.04
        accepted = (
            routed["macro_f1"] >= ELIGIBLE_ROUTED_F1
            and improved_seed_count >= 2
            and stable
            and class_share <= 0.90
        )
        aggregates[candidate]["eligibility"] = {
            "minimum_routed_macro_f1": ELIGIBLE_ROUTED_F1,
            "paired_improvement_by_seed": paired_improvements,
            "improved_seed_count": improved_seed_count,
            "genuine_nested_provenance": True,
            "seed_stability_guard": stable,
            "maximum_null_predicted_class_share": class_share,
            "class_collapse_guard": class_share <= 0.90,
            "eligible": accepted,
        }
        if accepted:
            eligible.append(candidate)
    selected = (
        max(eligible, key=lambda candidate: aggregates[candidate]["routed"]["macro_f1"])
        if eligible
        else None
    )
    pooled_i = np.concatenate(pooled[CANDIDATE_I])
    pooled_r = np.concatenate(pooled[CANDIDATE_R])
    weight_counts = Counter(
        str(record["candidate_v_selection"]["candidate_i_weight"])
        for record in completed
    )
    safe_folds = [
        {
            key: value
            for key, value in record.items()
            if key not in {"probabilities", "validation_positions"}
        }
        for record in completed
    ]
    return {
        "selection": {
            "selected_candidate": selected,
            "all_candidates_rejected": selected is None,
        },
        "aggregates": aggregates,
        "per_seed": seed_results,
        "paired_complementarity": {
            "pooled": _complementarity(repeated_null_truth, pooled_i, pooled_r),
            "per_seed": {
                str(seed): record
                for seed, record in zip(VALIDATION_SEEDS, complementarity, strict=True)
            },
        },
        "candidate_v_weight_distribution": dict(weight_counts),
        "fold_metrics": safe_folds,
        "validation": {
            "outer_seeds": list(VALIDATION_SEEDS),
            "outer_folds_per_seed": 5,
            "outer_evaluations": 15,
            "outer_workers": fold_workers,
            "start_method": "spawn",
            "zero_outer_group_overlap": True,
            "complete_outer_oof_coverage": True,
            "inner_folds_per_outer_evaluation": 3,
            "complete_inner_oof_coverage": True,
            "outer_validation_rows_used_by_inner_models": 0,
            "feature_configuration_sha256": fingerprint,
        },
        "raw_text_persisted": False,
        "row_level_probabilities_persisted": False,
    }
