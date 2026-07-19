"""Nested competition-metric evaluation for frozen V14 candidates Y0--Y4."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loading import validate_labeled_frame
from .v4_preprocessing import official_context_is_present
from .v5_lexical import build_v5_folds, deterministic_substring_prediction
from .v5_sparse import LOGISTIC_CONFIG, SparseNullModel
from .v7_runner import build_inner_splits
from .v8_route_complement import CANDIDATE_I, _build_v4a_classifier
from .v14_metrics import (
    metric_record,
    predictions_from_label1,
    select_absent_threshold_with_hard_present,
    select_route_thresholds,
    select_threshold,
)


CANDIDATES = ("Y0", "Y1", "Y2", "Y3", "Y4")
HISTORICAL = (
    "all_zero",
    "v4a_full",
    "substring_i",
    "substring_r",
    "substring_u",
    "v4a_i",
    "v4a_r",
    "v4a_u",
)
BOOTSTRAP_SEED = 141400
BOOTSTRAP_ITERATIONS = 10_000


def _r_pipeline() -> Pipeline:
    return Pipeline(
        [("scale", StandardScaler()), ("logistic", LogisticRegression(**LOGISTIC_CONFIG))]
    )


def _pipeline_probability(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    classes = list(model.named_steps["logistic"].classes_)
    return model.predict_proba(matrix)[:, classes.index(1)].astype(np.float64)


def _fit_base_probabilities(
    frame: pd.DataFrame,
    v4a_matrix: np.ndarray,
    r_matrix: np.ndarray,
    presence: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    train_absent = train[~presence[train]]
    validation_absent = validation[~presence[validation]]
    if set(labels[train_absent]) != {0, 1} or set(labels[validation_absent]) != {0, 1}:
        raise RuntimeError("V14 route-specific split lacks one official label")
    v4a = _build_v4a_classifier().fit(v4a_matrix[train], labels[train])
    probability_v4a = _pipeline_probability(v4a, v4a_matrix[validation])
    candidate_i = SparseNullModel(CANDIDATE_I).fit(
        frame.iloc[train_absent].reset_index(drop=True)
    )
    probability_i = candidate_i.predict_label1_probability(
        frame.iloc[validation_absent].reset_index(drop=True)
    )
    candidate_r = _r_pipeline().fit(r_matrix[train_absent], labels[train_absent])
    probability_r = _pipeline_probability(candidate_r, r_matrix[validation_absent])
    if not all(np.isfinite(values).all() for values in (probability_v4a, probability_i, probability_r)):
        raise RuntimeError("V14 produced NaN or Inf base probabilities")
    return probability_v4a, probability_i, probability_r


def _base_arrays(
    length: int,
    validation: np.ndarray,
    validation_presence: np.ndarray,
    probability_v4a: np.ndarray,
    probability_i: np.ndarray,
    probability_r: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {
        "v4a": np.full(length, np.nan, dtype=np.float64),
        "i": np.full(length, np.nan, dtype=np.float64),
        "r": np.full(length, np.nan, dtype=np.float64),
        "u": np.full(length, np.nan, dtype=np.float64),
    }
    result["v4a"][validation] = probability_v4a
    absent_validation = validation[~validation_presence]
    result["i"][absent_validation] = probability_i
    result["r"][absent_validation] = probability_r
    result["u"][absent_validation] = 0.5 * (probability_i + probability_r)
    return result


def _select_all(
    labels: np.ndarray,
    presence: np.ndarray,
    probabilities: dict[str, np.ndarray],
    substring: np.ndarray,
) -> dict[str, dict[str, Any]]:
    return {
        "Y0": select_threshold(labels, probabilities["v4a"])["selected"],
        "Y1": select_route_thresholds(labels, presence, probabilities["v4a"], probabilities["i"])[
            "selected"
        ],
        "Y2": select_route_thresholds(labels, presence, probabilities["v4a"], probabilities["r"])[
            "selected"
        ],
        "Y3": select_route_thresholds(labels, presence, probabilities["v4a"], probabilities["u"])[
            "selected"
        ],
        "Y4": select_absent_threshold_with_hard_present(
            labels, presence, substring, probabilities["i"]
        )["selected"],
    }


def _apply_candidate(
    candidate: str,
    presence: np.ndarray,
    probabilities: dict[str, np.ndarray],
    substring: np.ndarray,
    thresholds: dict[str, Any],
) -> np.ndarray:
    prediction = np.empty(len(presence), dtype=np.int64)
    if candidate == "Y0":
        return predictions_from_label1(probabilities["v4a"], thresholds["threshold"])
    if candidate == "Y4":
        prediction[presence] = substring[presence]
        prediction[~presence] = predictions_from_label1(
            probabilities["i"][~presence], thresholds["absent_threshold"]
        )
        return prediction
    absent_key = {"Y1": "i", "Y2": "r", "Y3": "u"}[candidate]
    prediction[presence] = predictions_from_label1(
        probabilities["v4a"][presence], thresholds["present_threshold"]
    )
    prediction[~presence] = predictions_from_label1(
        probabilities[absent_key][~presence], thresholds["absent_threshold"]
    )
    return prediction


def _historical_predictions(
    presence: np.ndarray, probabilities: dict[str, np.ndarray], substring: np.ndarray
) -> dict[str, np.ndarray]:
    v4a = predictions_from_label1(probabilities["v4a"], 0.50)
    result = {"all_zero": np.zeros(len(presence), dtype=np.int64), "v4a_full": v4a}
    for prefix, present_prediction in (("substring", substring), ("v4a", v4a)):
        for suffix in ("i", "r", "u"):
            prediction = present_prediction.copy()
            prediction[~presence] = predictions_from_label1(probabilities[suffix][~presence], 0.50)
            result[f"{prefix}_{suffix}"] = prediction
    return result


def _index_hash(values: np.ndarray) -> str:
    return hashlib.sha256(",".join(map(str, values.tolist())).encode("ascii")).hexdigest()


def _bootstrap_interval(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    differences = np.empty(BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        differences[iteration] = (
            metric_record(truth[indices], candidate[indices])["label0_f1"]
            - metric_record(truth[indices], baseline[indices])["label0_f1"]
        )
    return {
        "seed": BOOTSTRAP_SEED,
        "iterations": BOOTSTRAP_ITERATIONS,
        "difference_95_percent_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def run_v14_nested(
    frame: pd.DataFrame, v4a_matrix: np.ndarray, candidate_r_matrix: np.ndarray
) -> dict[str, Any]:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    labels = validated["label"].to_numpy(dtype=np.int64)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if (int(presence.sum()), int((~presence).sum())) != (130, 169):
        raise RuntimeError("V14 corrected route totals differ from 130/169")
    if v4a_matrix.shape[0] != len(validated) or candidate_r_matrix.shape != (len(validated), candidate_r_matrix.shape[1]):
        raise ValueError("V14 feature matrices are not row-aligned")
    if not np.isfinite(v4a_matrix).all() or not np.isfinite(candidate_r_matrix[~presence]).all():
        raise ValueError("V14 feature matrices contain NaN or Inf")
    substring = np.asarray(
        [deterministic_substring_prediction(row.context, row.response_bn) for row in validated.itertuples(index=False)],
        dtype=np.int64,
    )
    folds = build_v5_folds(validated)
    groups = np.asarray(folds.audit.group_ids, dtype=object)
    outer_records: list[dict[str, Any]] = []
    seed_nested = {seed: {name: np.full(len(validated), -1, dtype=np.int64) for name in CANDIDATES} for seed in folds.seeds}
    seed_historical = {seed: {name: np.full(len(validated), -1, dtype=np.int64) for name in HISTORICAL} for seed in folds.seeds}
    fold_improvement_reference: list[dict[str, Any]] = []
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        if set(train) & set(validation) or set(groups[train]) & set(groups[validation]):
            raise RuntimeError("V14 outer split overlap detected")
        inner_splits = build_inner_splits(labels[train], groups[train], outer_seed=split.seed, outer_fold=split.fold)
        inner = {key: np.full(train.size, np.nan, dtype=np.float64) for key in ("v4a", "i", "r", "u")}
        coverage = np.zeros(train.size, dtype=np.int64)
        inner_records = []
        for inner_fold, (inner_train_local, inner_validation_local) in enumerate(inner_splits, start=1):
            inner_train = train[inner_train_local]
            inner_validation = train[inner_validation_local]
            pv, pi, pr = _fit_base_probabilities(
                validated, v4a_matrix, candidate_r_matrix, presence, inner_train, inner_validation
            )
            local = _base_arrays(train.size, inner_validation_local, presence[inner_validation], pv, pi, pr)
            for key in inner:
                mask = np.isfinite(local[key])
                inner[key][mask] = local[key][mask]
            coverage[inner_validation_local] += 1
            inner_records.append(
                {
                    "inner_fold": inner_fold,
                    "training_index_sha256": _index_hash(inner_train),
                    "validation_index_sha256": _index_hash(inner_validation),
                    "row_overlap_count": 0,
                    "group_overlap_count": 0,
                    "outer_validation_rows_used": 0,
                }
            )
        if not np.all(coverage == 1) or not np.isfinite(inner["v4a"]).all() or any(
            not np.isfinite(inner[key][~presence[train]]).all() for key in ("i", "r", "u")
        ):
            raise RuntimeError("V14 inner OOF coverage is incomplete")
        selected = _select_all(labels[train], presence[train], inner, substring[train])
        pv, pi, pr = _fit_base_probabilities(
            validated, v4a_matrix, candidate_r_matrix, presence, train, validation
        )
        outer = _base_arrays(len(validated), validation, presence[validation], pv, pi, pr)
        outer_predictions = {}
        for candidate in CANDIDATES:
            local_probabilities = {key: values[validation] for key, values in outer.items()}
            prediction = _apply_candidate(
                candidate, presence[validation], local_probabilities, substring[validation], selected[candidate]
            )
            seed_nested[split.seed][candidate][validation] = prediction
            outer_predictions[candidate] = prediction
        historical = _historical_predictions(
            presence[validation], {key: values[validation] for key, values in outer.items()}, substring[validation]
        )
        for name, prediction in historical.items():
            seed_historical[split.seed][name][validation] = prediction
        baseline_fold_f1 = metric_record(labels[validation], historical["v4a_full"])["label0_f1"]
        outer_records.append(
            {
                "seed": split.seed,
                "fold": split.fold,
                "training_rows": int(train.size),
                "validation_rows": int(validation.size),
                "training_groups": int(len(set(groups[train]))),
                "validation_groups": int(len(set(groups[validation]))),
                "row_overlap_count": 0,
                "group_overlap_count": 0,
                "inner": inner_records,
                "selected_thresholds": selected,
                "metrics": {name: metric_record(labels[validation], prediction) for name, prediction in outer_predictions.items()},
            }
        )
        fold_improvement_reference.append(
            {name: metric_record(labels[validation], prediction)["label0_f1"] - baseline_fold_f1 for name, prediction in outer_predictions.items()}
        )
    repeated_truth = np.tile(labels, len(folds.seeds))
    repeated_groups = np.concatenate(
        [np.asarray([f"{seed}:{value}" for value in groups], dtype=object) for seed in folds.seeds]
    )
    historical_aggregate = {}
    historical_pooled_predictions = {}
    for name in HISTORICAL:
        blocks = [seed_historical[seed][name] for seed in folds.seeds]
        if any(np.any(block < 0) for block in blocks):
            raise RuntimeError("V14 historical OOF coverage is incomplete")
        pooled = np.concatenate(blocks)
        historical_pooled_predictions[name] = pooled
        historical_aggregate[name] = {
            **metric_record(repeated_truth, pooled),
            "per_seed": {str(seed): metric_record(labels, seed_historical[seed][name]) for seed in folds.seeds},
        }
    best_baseline_name = max(HISTORICAL, key=lambda name: historical_aggregate[name]["label0_f1"])
    best_baseline = historical_pooled_predictions[best_baseline_name]
    nested_aggregate = {}
    for candidate in CANDIDATES:
        blocks = [seed_nested[seed][candidate] for seed in folds.seeds]
        if any(np.any(block < 0) for block in blocks):
            raise RuntimeError("V14 nested OOF coverage is incomplete")
        pooled = np.concatenate(blocks)
        per_seed = {str(seed): metric_record(labels, seed_nested[seed][candidate]) for seed in folds.seeds}
        seed_values = np.asarray([record["label0_f1"] for record in per_seed.values()])
        improvements = {
            str(seed): per_seed[str(seed)]["label0_f1"] - historical_aggregate[best_baseline_name]["per_seed"][str(seed)]["label0_f1"]
            for seed in folds.seeds
        }
        fold_improvements = [record[candidate] for record in fold_improvement_reference]
        threshold_records = [record["selected_thresholds"][candidate] for record in outer_records]
        nested_aggregate[candidate] = {
            **metric_record(repeated_truth, pooled),
            "per_seed": per_seed,
            "seed_distribution": {
                "mean": float(seed_values.mean()), "std": float(seed_values.std(ddof=0)),
                "minimum": float(seed_values.min()), "maximum": float(seed_values.max()),
            },
            "improvement_by_seed": improvements,
            "improved_seed_count": int(sum(value > 0 for value in improvements.values())),
            "improved_outer_fold_count": int(sum(value > 0 for value in fold_improvements)),
            "threshold_distribution": dict(Counter(
                str({key: value for key, value in record.items() if key.endswith("threshold")})
                for record in threshold_records
            )),
            "bootstrap_vs_best_fixed_050": _bootstrap_interval(
                repeated_truth, pooled, best_baseline, repeated_groups
            ),
            "complete_nested_provenance": True,
            "zero_group_overlap": True,
            "threshold_selection_leakage": False,
        }
    eligible = [
        candidate for candidate in CANDIDATES
        if nested_aggregate[candidate]["improved_seed_count"] >= 2
        and max(nested_aggregate[candidate]["predicted_label0_share"], 1.0 - nested_aggregate[candidate]["predicted_label0_share"]) <= 0.90
    ]
    selected = max(eligible, key=lambda candidate: nested_aggregate[candidate]["label0_f1"]) if eligible else None
    improvement = (
        nested_aggregate[selected]["label0_f1"] - historical_aggregate[best_baseline_name]["label0_f1"]
        if selected else None
    )
    return {
        "historical_threshold_050": historical_aggregate,
        "best_threshold_050_baseline": best_baseline_name,
        "nested": nested_aggregate,
        "outer_folds": outer_records,
        "selection": {
            "selected_candidate": selected,
            "improvement": improvement,
            "submission_worthy": bool(selected and improvement is not None and improvement >= 0.01),
            "strong_improvement": bool(selected and improvement is not None and improvement >= 0.03),
        },
        "validation": {
            "seeds": list(folds.seeds), "outer_folds_per_seed": 5, "inner_folds": 3,
            "outer_evaluations": 15, "row_overlap_count": 0, "group_overlap_count": 0,
        },
    }
