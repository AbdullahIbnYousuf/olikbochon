"""Frozen grouped-validation design and low-capacity lexical candidates for V5."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loading import DataValidationError, validate_labeled_frame
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v3_data import GroupAudit, build_official_groups
from .v4_diagnostics import probability_diagnostics
from .v4_preprocessing import official_context_is_present
from .v4_validation import (
    FOLD_COUNT,
    VALIDATION_SEEDS,
    RepeatedGroupedFolds,
    make_repeated_grouped_folds,
)
from .v5_features import (
    CONTEXT_ABSENT_FEATURES,
    CONTEXT_PRESENT_FEATURES,
    V5FeatureExtractor,
    aggregate_feature_audit,
)
from .v5_normalization import normalize_v5


EXPERIMENT_NAME = "v5_official_lexical_baseline"
CANDIDATE_SET = "frozen"
CANDIDATES = (
    "deterministic_substring_rule",
    "context_present_logistic",
    "context_absent_logistic",
    "routed_lexical_logistic",
)
LOGISTIC_RANDOM_STATE = 42
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITERATIONS = 2000
LOGISTIC_SOLVER = "liblinear"


@dataclass(frozen=True)
class V5GroupAudit(GroupAudit):
    """Existing V3 graph plus exact V5 row, prompt, and present-context identities."""

    inherited_group_links: int
    v5_exact_row_links: int
    prompt_identity_links: int
    context_identity_links: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        first, second = self.find(left), self.find(right)
        if first != second:
            low, high = sorted((first, second))
            self.parent[high] = low


def _fingerprint(*values: str) -> str:
    return hashlib.sha256("\n\x1f\n".join(values).encode("utf-8")).hexdigest()


def _link_members(union: _UnionFind, groups: dict[str, list[int]]) -> int:
    links = 0
    for members in groups.values():
        for right in members[1:]:
            union.union(members[0], right)
            links += 1
    return links


def build_v5_groups(frame: pd.DataFrame) -> V5GroupAudit:
    """Strengthen V3 families without grouping every absent-context row together."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    routed = validated.copy()
    routed["context"] = [
        value if official_context_is_present(value) else None for value in routed["context"]
    ]
    inherited = build_official_groups(routed)
    union = _UnionFind(len(validated))

    inherited_groups: dict[str, list[int]] = {}
    exact_groups: dict[str, list[int]] = {}
    prompt_groups: dict[str, list[int]] = {}
    context_groups: dict[str, list[int]] = {}
    labels = validated["label"].to_numpy(dtype=np.int64)

    for index, row in enumerate(validated.itertuples(index=False)):
        present = official_context_is_present(row.context)
        prompt = normalize_v5(row.prompt_bn)
        context = normalize_v5(row.context) if present else ""
        response = normalize_v5(row.response_bn)
        inherited_groups.setdefault(inherited.group_ids[index], []).append(index)
        exact_groups.setdefault(_fingerprint(prompt, context, response), []).append(index)
        prompt_groups.setdefault(_fingerprint(prompt), []).append(index)
        if present:
            context_groups.setdefault(_fingerprint(context), []).append(index)

    conflicts = sum(
        len({int(labels[index]) for index in members}) > 1
        for members in exact_groups.values()
    )
    if conflicts:
        raise DataValidationError(
            f"Found {conflicts} V5-normalized exact rows with conflicting labels"
        )

    inherited_links = _link_members(union, inherited_groups)
    exact_links = _link_members(union, exact_groups)
    prompt_links = _link_members(union, prompt_groups)
    context_links = _link_members(union, context_groups)
    roots = [union.find(index) for index in range(len(validated))]
    root_order = {root: position for position, root in enumerate(sorted(set(roots)))}
    group_ids = tuple(f"v5-group-{root_order[root]:04d}" for root in roots)
    counts = pd.Series(group_ids).value_counts()
    return V5GroupAudit(
        group_ids=group_ids,
        group_count=int(counts.size),
        nontrivial_groups=int((counts > 1).sum()),
        largest_group=int(counts.max()),
        exact_duplicate_pairs=inherited.exact_duplicate_pairs,
        prompt_context_pairs=inherited.prompt_context_pairs,
        near_duplicate_pairs=inherited.near_duplicate_pairs,
        conflicting_exact_texts=conflicts,
        inherited_group_links=inherited_links,
        v5_exact_row_links=exact_links,
        prompt_identity_links=prompt_links,
        context_identity_links=context_links,
    )


def build_v5_folds(frame: pd.DataFrame) -> RepeatedGroupedFolds:
    """Build the frozen repeated folds with no naive-random fallback."""
    audit = build_v5_groups(frame)
    return make_repeated_grouped_folds(
        frame["label"].to_numpy(dtype=np.int64),
        audit,
        seeds=VALIDATION_SEEDS,
        fold_count=FOLD_COUNT,
    )


def fold_feasibility_audit(
    frame: pd.DataFrame, folds: RepeatedGroupedFolds
) -> dict[str, Any]:
    """Verify both classes in both schema routes for every train/validation split."""
    labels = frame["label"].to_numpy(dtype=np.int64)
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    groups = np.asarray(folds.audit.group_ids, dtype=object)
    records: list[dict[str, Any]] = []
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        overlap = set(groups[train]) & set(groups[validation])
        if overlap:
            raise RuntimeError(f"V5 group overlap in seed {split.seed}, fold {split.fold}")
        record: dict[str, Any] = {
            "seed": split.seed,
            "fold": split.fold,
            "train_rows": int(train.size),
            "validation_rows": int(validation.size),
            "train_groups": int(len(set(groups[train]))),
            "validation_groups": int(len(set(groups[validation]))),
            "group_overlap_count": 0,
            "routes": {},
        }
        for route_name, route_mask in (
            ("context_present", presence),
            ("context_absent", ~presence),
        ):
            route_record: dict[str, Any] = {}
            for split_name, indices in (("train", train), ("validation", validation)):
                selected = labels[indices][route_mask[indices]]
                counts = np.bincount(selected, minlength=2)
                if set(np.unique(selected)) != {0, 1}:
                    raise RuntimeError(
                        f"{route_name} {split_name} lacks a class for seed "
                        f"{split.seed}, fold {split.fold}"
                    )
                route_record[split_name] = {
                    "rows": int(selected.size),
                    "label_0": int(counts[0]),
                    "label_1": int(counts[1]),
                }
            record["routes"][route_name] = route_record
        records.append(record)
    return {
        "feasible": True,
        "seeds": list(folds.seeds),
        "fold_count_per_seed": FOLD_COUNT,
        "total_folds": len(folds.folds),
        "zero_group_overlap": True,
        "all_route_splits_have_both_labels": True,
        "folds": records,
    }


def build_logistic_pipeline() -> Pipeline:
    """Return the exact frozen standardized logistic configuration."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    C=LOGISTIC_C,
                    max_iter=LOGISTIC_MAX_ITERATIONS,
                    random_state=LOGISTIC_RANDOM_STATE,
                    solver=LOGISTIC_SOLVER,
                ),
            ),
        ]
    )


def deterministic_substring_prediction(context: Any, response: Any) -> int:
    """Return faithful only for a nonempty exact normalized response substring."""
    normalized_context = normalize_v5(context)
    normalized_response = normalize_v5(response)
    return int(bool(normalized_response) and normalized_response in normalized_context)


def majority_fallback(labels: Sequence[int]) -> int:
    """Choose the training-fold absent-route majority, breaking exact ties toward label 0."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    return int(1 if counts[1] > counts[0] else 0)


def _label1_probability(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    logistic = model.named_steps["logistic"]
    classes = list(logistic.classes_)
    return model.predict_proba(matrix)[:, classes.index(1)].astype(np.float64)


def _prediction_counts(predictions: np.ndarray) -> dict[str, int]:
    counts = np.bincount(predictions, minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def _metric_record(
    truth: np.ndarray,
    predictions: np.ndarray,
    presence: np.ndarray | None = None,
) -> dict[str, Any]:
    record = {
        **classification_metrics(truth, predictions),
        "prediction_counts": _prediction_counts(predictions),
    }
    if presence is not None:
        record["route_metrics"] = subgroup_metrics(truth, predictions, presence)
    return record


def _coefficient_record(
    model: Pipeline, names: Sequence[str], *, seed: int, fold: int, route: str
) -> dict[str, Any]:
    coefficients = model.named_steps["logistic"].coef_[0]
    return {
        "seed": seed,
        "fold": fold,
        "route": route,
        "standardized_coefficients": {
            name: float(value) for name, value in zip(names, coefficients)
        },
    }


def _coefficient_summary(records: Sequence[dict[str, Any]], route: str) -> dict[str, Any]:
    selected = [record for record in records if record["route"] == route]
    names = (
        CONTEXT_PRESENT_FEATURES if route == "context_present" else CONTEXT_ABSENT_FEATURES
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        values = np.asarray(
            [record["standardized_coefficients"][name] for record in selected],
            dtype=np.float64,
        )
        rows.append(
            {
                "feature": name,
                "mean_standardized_coefficient": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "positive_fraction": float((values > 0).mean()),
                "negative_fraction": float((values < 0).mean()),
                "sign_stable": bool((values > 0).all() or (values < 0).all()),
            }
        )
    ranked = sorted(rows, key=lambda row: row["mean_standardized_coefficient"])
    return {
        "fold_count": len(selected),
        "features": rows,
        "top_negative": ranked[:5],
        "top_positive": list(reversed(ranked[-5:])),
        "interpretation_warning": "Do not overinterpret coefficients with unstable signs.",
    }


def _safe_probability_diagnostics(truth: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    try:
        return probability_diagnostics(truth, probabilities)
    except ValueError as exc:
        predictions = predictions_from_label1(probabilities, 0.5)
        return {
            "row_count": int(truth.size),
            "brier_score": float(np.mean(np.square(probabilities - truth))),
            "threshold_050": _metric_record(truth, predictions),
            "frozen_grid_eligible": False,
            "frozen_grid_error": str(exc),
        }


def _probability_diagnostics_with_routes(
    truth: np.ndarray,
    probabilities: np.ndarray,
    presence: np.ndarray | None = None,
) -> dict[str, Any]:
    diagnostics = _safe_probability_diagnostics(truth, probabilities)
    selected = diagnostics.get("best_frozen_grid")
    if selected is not None:
        threshold = float(selected["threshold"])
        predictions = predictions_from_label1(probabilities, threshold)
        selected["passes_class_collapse_guard"] = bool(
            selected["maximum_predicted_class_share"] <= 0.90
        )
        if presence is not None:
            selected["route_metrics"] = subgroup_metrics(truth, predictions, presence)
    return diagnostics


def _seed_metric_distribution(seed_results: dict[str, Any]) -> dict[str, Any]:
    fields = ("macro_f1", "f1_label0", "f1_label1", "accuracy")
    result: dict[str, Any] = {}
    for candidate in ("candidate_a", "candidate_b", "candidate_c", "candidate_d"):
        result[candidate] = {}
        for field in fields:
            values = np.asarray(
                [record[candidate][field] for record in seed_results.values()],
                dtype=np.float64,
            )
            result[candidate][field] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return result


def _record_metric_distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
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


def run_frozen_lexical_experiment(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate all four candidates in memory and return aggregate/fold-level records only."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    folds = build_v5_folds(validated)
    feasibility = fold_feasibility_audit(validated, folds)
    truth = validated["label"].to_numpy(dtype=np.int64)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    coefficient_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    seed_results: dict[str, Any] = {}
    pooled_substring_predictions: list[np.ndarray] = []
    pooled_present_probabilities: list[np.ndarray] = []
    pooled_absent_probabilities: list[np.ndarray] = []
    pooled_routed_probabilities: list[np.ndarray] = []

    for seed in folds.seeds:
        substring_oof = np.full(len(validated), -1, dtype=np.int64)
        present_probabilities = np.full(len(validated), np.nan, dtype=np.float64)
        absent_probabilities = np.full(len(validated), np.nan, dtype=np.float64)
        for split in (item for item in folds.folds if item.seed == seed):
            train = np.asarray(split.train_indices, dtype=np.int64)
            validation = np.asarray(split.validation_indices, dtype=np.int64)
            train_present = train[presence[train]]
            train_absent = train[~presence[train]]
            validation_present = validation[presence[validation]]
            validation_absent = validation[~presence[validation]]

            present_extractor = V5FeatureExtractor.fit(validated.iloc[train_present])
            absent_extractor = V5FeatureExtractor.fit(validated.iloc[train_absent])
            present_model = build_logistic_pipeline()
            absent_model = build_logistic_pipeline()
            present_model.fit(
                present_extractor.matrix(validated, train_present, context_present=True),
                truth[train_present],
            )
            absent_model.fit(
                absent_extractor.matrix(validated, train_absent, context_present=False),
                truth[train_absent],
            )
            present_probabilities[validation_present] = _label1_probability(
                present_model,
                present_extractor.matrix(
                    validated, validation_present, context_present=True
                ),
            )
            absent_probabilities[validation_absent] = _label1_probability(
                absent_model,
                absent_extractor.matrix(
                    validated, validation_absent, context_present=False
                ),
            )
            fallback = majority_fallback(truth[train_absent])
            for index in validation:
                row = validated.iloc[int(index)]
                substring_oof[index] = (
                    deterministic_substring_prediction(row.context, row.response_bn)
                    if presence[index]
                    else fallback
                )
            present_predictions = predictions_from_label1(
                present_probabilities[validation_present], 0.5
            )
            absent_predictions = predictions_from_label1(
                absent_probabilities[validation_absent], 0.5
            )
            present_diagnostics = _probability_diagnostics_with_routes(
                truth[validation_present], present_probabilities[validation_present]
            )
            absent_diagnostics = _probability_diagnostics_with_routes(
                truth[validation_absent], absent_probabilities[validation_absent]
            )
            routed_predictions = np.empty(validation.size, dtype=np.int64)
            routed_predictions[presence[validation]] = present_predictions
            routed_predictions[~presence[validation]] = absent_predictions
            routed_probabilities = np.where(
                presence[validation],
                present_probabilities[validation],
                absent_probabilities[validation],
            )
            routed_diagnostics = _probability_diagnostics_with_routes(
                truth[validation], routed_probabilities, presence[validation]
            )
            fold_records.append(
                {
                    "seed": seed,
                    "fold": split.fold,
                    "train_rows": int(train.size),
                    "validation_rows": int(validation.size),
                    "group_overlap_count": 0,
                    "candidate_a": _metric_record(
                        truth[validation], substring_oof[validation], presence[validation]
                    ),
                    "candidate_b": {
                        **_metric_record(truth[validation_present], present_predictions),
                        "probability_diagnostics": present_diagnostics,
                    },
                    "candidate_c": {
                        **_metric_record(truth[validation_absent], absent_predictions),
                        "probability_diagnostics": absent_diagnostics,
                    },
                    "candidate_d": {
                        **_metric_record(
                            truth[validation], routed_predictions, presence[validation]
                        ),
                        "probability_diagnostics": routed_diagnostics,
                    },
                    "absent_majority_fallback_label": fallback,
                }
            )
            coefficient_records.extend(
                (
                    _coefficient_record(
                        present_model,
                        CONTEXT_PRESENT_FEATURES,
                        seed=seed,
                        fold=split.fold,
                        route="context_present",
                    ),
                    _coefficient_record(
                        absent_model,
                        CONTEXT_ABSENT_FEATURES,
                        seed=seed,
                        fold=split.fold,
                        route="context_absent",
                    ),
                )
            )

        if (substring_oof < 0).any() or not np.isfinite(
            np.where(presence, present_probabilities, absent_probabilities)
        ).all():
            raise RuntimeError(f"Incomplete V5 OOF coverage for seed {seed}")
        routed_probabilities = np.where(presence, present_probabilities, absent_probabilities)
        routed_predictions = predictions_from_label1(routed_probabilities, 0.5)
        pooled_substring_predictions.append(substring_oof)
        pooled_present_probabilities.append(present_probabilities[presence])
        pooled_absent_probabilities.append(absent_probabilities[~presence])
        pooled_routed_probabilities.append(routed_probabilities)
        seed_results[str(seed)] = {
            "complete_oof_coverage": True,
            "candidate_a": _metric_record(truth, substring_oof, presence),
            "candidate_b": {
                **_metric_record(
                    truth[presence],
                    predictions_from_label1(present_probabilities[presence], 0.5),
                ),
                "probability_diagnostics": _probability_diagnostics_with_routes(
                    truth[presence], present_probabilities[presence]
                ),
            },
            "candidate_c": {
                **_metric_record(
                    truth[~presence],
                    predictions_from_label1(absent_probabilities[~presence], 0.5),
                ),
                "probability_diagnostics": _probability_diagnostics_with_routes(
                    truth[~presence], absent_probabilities[~presence]
                ),
            },
            "candidate_d": {
                **_metric_record(truth, routed_predictions, presence),
                "probability_diagnostics": _probability_diagnostics_with_routes(
                    truth, routed_probabilities, presence
                ),
            },
        }

    repeated_truth = np.tile(truth, len(folds.seeds))
    repeated_presence = np.tile(presence, len(folds.seeds))
    repeated_present_truth = np.tile(truth[presence], len(folds.seeds))
    repeated_absent_truth = np.tile(truth[~presence], len(folds.seeds))
    aggregate = {
        "pooled_repeated_oof_role": (
            "Each official row appears once per seed; pooled counts are repeated OOF, not "
            "independent rows."
        ),
        "candidate_a": _metric_record(
            repeated_truth,
            np.concatenate(pooled_substring_predictions),
            repeated_presence,
        ),
        "candidate_b": {
            **_metric_record(
                repeated_present_truth,
                predictions_from_label1(
                    np.concatenate(pooled_present_probabilities), 0.5
                ),
            ),
            "probability_diagnostics": _probability_diagnostics_with_routes(
                repeated_present_truth,
                np.concatenate(pooled_present_probabilities),
            ),
        },
        "candidate_c": {
            **_metric_record(
                repeated_absent_truth,
                predictions_from_label1(
                    np.concatenate(pooled_absent_probabilities), 0.5
                ),
            ),
            "probability_diagnostics": _probability_diagnostics_with_routes(
                repeated_absent_truth,
                np.concatenate(pooled_absent_probabilities),
            ),
        },
        "candidate_d": {
            **_metric_record(
                repeated_truth,
                predictions_from_label1(
                    np.concatenate(pooled_routed_probabilities), 0.5
                ),
                repeated_presence,
            ),
            "probability_diagnostics": _probability_diagnostics_with_routes(
                repeated_truth,
                np.concatenate(pooled_routed_probabilities),
                repeated_presence,
            ),
        },
        "seed_metric_distribution": _seed_metric_distribution(seed_results),
    }
    selected_d = aggregate["candidate_d"]["probability_diagnostics"].get(
        "best_frozen_grid"
    )
    if selected_d is not None:
        selected_threshold = float(selected_d["threshold"])
        selected_seed_records = [
            _metric_record(
                truth,
                predictions_from_label1(probabilities, selected_threshold),
                presence,
            )
            for probabilities in pooled_routed_probabilities
        ]
        selected_d["seed_metric_distribution"] = _record_metric_distribution(
            selected_seed_records
        )
        selected_d["threshold_gain_over_050"] = float(
            selected_d["macro_f1"] - aggregate["candidate_d"]["macro_f1"]
        )

    return {
        "experiment": EXPERIMENT_NAME,
        "candidate_set": CANDIDATE_SET,
        "candidates": list(CANDIDATES),
        "labels": {"0": "hallucinated", "1": "faithful"},
        "validation": feasibility,
        "feature_audit": aggregate_feature_audit(validated),
        "fold_metrics": fold_records,
        "per_seed": seed_results,
        "aggregate": aggregate,
        "fold_coefficients": coefficient_records,
        "coefficient_reports": {
            "context_present": _coefficient_summary(coefficient_records, "context_present"),
            "context_absent": _coefficient_summary(coefficient_records, "context_absent"),
        },
        "row_level_values_persisted": False,
    }
