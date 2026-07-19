"""Repeated group-isolated validation and frozen Version 4 selection rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v3_data import GroupAudit, build_official_groups
from .v4_preprocessing import official_context_is_present


VALIDATION_SEEDS = (17, 29, 43)
FOLD_COUNT = 5
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(20, 81, 2))
CANDIDATE_ORDER = (
    "v3_compatible_baseline",
    "structured_serialization",
    "routed_serialization",
    "selected_serialization_field_aware",
)


class ValidationFeasibilityError(RuntimeError):
    """Raised rather than falling back to a split that can leak groups."""


@dataclass(frozen=True)
class ValidationFold:
    seed: int
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


@dataclass(frozen=True)
class RepeatedGroupedFolds:
    audit: GroupAudit
    seeds: tuple[int, ...]
    folds: tuple[ValidationFold, ...]
    split_strategy: str = "repeated_stratified_group_kfold"


@dataclass(frozen=True)
class CandidateEvaluation:
    name: str
    macro_f1_mean: float
    macro_f1_std: float
    f1_label0_mean: float
    f1_label1_mean: float
    maximum_predicted_class_share: float


def make_repeated_grouped_folds(
    labels: Sequence[int],
    audit: GroupAudit,
    *,
    seeds: Sequence[int] = VALIDATION_SEEDS,
    fold_count: int = FOLD_COUNT,
) -> RepeatedGroupedFolds:
    """Create validated repeated folds while keeping every family isolated."""
    truth = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(audit.group_ids, dtype=object)
    if truth.size != groups.size or truth.size == 0:
        raise ValidationFeasibilityError("Labels and group IDs must be nonempty and aligned")
    if not np.isin(truth, [0, 1]).all():
        raise ValidationFeasibilityError("Labels must be binary")
    if len(set(groups)) < fold_count:
        raise ValidationFeasibilityError(
            f"Only {len(set(groups))} groups are available for {fold_count} folds"
        )
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Validation seeds must be nonempty and unique")

    rows = np.arange(truth.size)
    result: list[ValidationFold] = []
    for seed in normalized_seeds:
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        assignments = np.zeros(truth.size, dtype=np.int64)
        try:
            split_iterator = splitter.split(rows, truth, groups)
            for fold, (train, validation) in enumerate(split_iterator, start=1):
                if set(groups[train]) & set(groups[validation]):
                    raise ValidationFeasibilityError(
                        f"Group leakage detected for seed {seed}, fold {fold}"
                    )
                if set(np.unique(truth[validation])) != {0, 1}:
                    raise ValidationFeasibilityError(
                        f"Seed {seed}, fold {fold} does not contain both classes"
                    )
                assignments[validation] += 1
                result.append(
                    ValidationFold(
                        seed,
                        fold,
                        tuple(int(value) for value in train),
                        tuple(int(value) for value in validation),
                    )
                )
        except ValueError as exc:
            raise ValidationFeasibilityError(
                f"Grouped split is infeasible for seed {seed}: {exc}"
            ) from exc
        if not np.all(assignments == 1):
            raise ValidationFeasibilityError(
                f"Seed {seed} did not validate every row exactly once"
            )
    if len(result) != len(normalized_seeds) * fold_count:
        raise ValidationFeasibilityError("Repeated grouped split returned an unexpected fold count")
    return RepeatedGroupedFolds(audit, normalized_seeds, tuple(result))


def build_repeated_official_folds(frame: Any) -> RepeatedGroupedFolds:
    """Build V4 folds with official sentinel-aware context grouping."""
    routed_frame = frame.copy()
    routed_frame["context"] = [
        value if official_context_is_present(value) else None
        for value in routed_frame["context"]
    ]
    audit = build_official_groups(routed_frame)
    return make_repeated_grouped_folds(frame["label"].to_numpy(dtype=np.int64), audit)


def fold_metric_record(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    has_context: Sequence[bool],
    *,
    seed: int,
    fold: int,
) -> dict[str, Any]:
    """Return the frozen aggregate metrics for one validation fold."""
    base = classification_metrics(y_true, y_pred)
    regimes = subgroup_metrics(y_true, y_pred, has_context)
    return {
        "seed": int(seed),
        "fold": int(fold),
        **base,
        "context_present_macro_f1": regimes["context_present"].get("macro_f1"),
        "null_context_macro_f1": regimes["context_absent"].get("macro_f1"),
        "context_metrics": regimes,
    }


def summarize_fold_metrics(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate required scalar metrics across folds and repeats."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one fold metric record is required")
    fields = (
        "macro_f1",
        "f1_label0",
        "f1_label1",
        "accuracy",
        "context_present_macro_f1",
        "null_context_macro_f1",
    )
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = np.asarray([row[field] for row in rows if row[field] is not None], dtype=float)
        if not values.size:
            raise ValueError(f"No fold supplied {field}")
        summary[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def select_threshold(
    truth: Sequence[int], probabilities_label1: Sequence[float]
) -> tuple[float, dict[str, Any]]:
    """Select on the frozen grid using macro F1 and predeclared tie-breakers."""
    ranked: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
    for threshold in THRESHOLD_GRID:
        predictions = predictions_from_label1(probabilities_label1, threshold)
        metrics = classification_metrics(truth, predictions)
        class_share = float(np.bincount(predictions, minlength=2).max() / len(predictions))
        if class_share > 0.90:
            continue
        rank = (
            float(metrics["macro_f1"]),
            float(metrics["f1_label0"]),
            -abs(threshold - 0.5),
            -threshold,
        )
        ranked.append((rank, threshold, {**metrics, "maximum_predicted_class_share": class_share}))
    if not ranked:
        raise ValueError("Every threshold candidate violates the class-collapse guard")
    _, threshold, metrics = max(ranked, key=lambda item: item[0])
    return threshold, metrics


def route_threshold_diagnostics(
    truth: Sequence[int],
    probabilities_label1: Sequence[float],
    has_context: Sequence[bool],
) -> dict[str, Any]:
    """Report route-optimal frozen-grid thresholds for diagnosis only, never deployment."""
    labels = np.asarray(truth, dtype=np.int64)
    probabilities = np.asarray(probabilities_label1, dtype=np.float64)
    presence = np.asarray(has_context, dtype=bool)
    if not (labels.size == probabilities.size == presence.size):
        raise ValueError("Route diagnostic inputs must be aligned")
    result: dict[str, Any] = {
        "role": "diagnostic_only_not_selected_or_deployed",
        "routes": {},
    }
    for name, mask in (("context_present", presence), ("context_absent", ~presence)):
        if not mask.any() or set(np.unique(labels[mask])) != {0, 1}:
            raise ValueError(f"Route {name} must be nonempty and contain both labels")
        ranked: list[tuple[tuple[float, float, float, float], float, dict[str, Any]]] = []
        for threshold in THRESHOLD_GRID:
            predictions = predictions_from_label1(probabilities[mask], threshold)
            metrics = classification_metrics(labels[mask], predictions)
            class_share = float(
                np.bincount(predictions, minlength=2).max() / len(predictions)
            )
            rank = (
                float(metrics["macro_f1"]),
                float(metrics["f1_label0"]),
                -abs(threshold - 0.5),
                -threshold,
            )
            ranked.append(
                (
                    rank,
                    threshold,
                    {**metrics, "maximum_predicted_class_share": class_share},
                )
            )
        _, threshold, metrics = max(ranked, key=lambda item: item[0])
        predictions = predictions_from_label1(probabilities[mask], threshold)
        counts = np.bincount(predictions, minlength=2)
        result["routes"][name] = {
            "row_count": int(mask.sum()),
            "threshold": threshold,
            "metrics": metrics,
            "passes_class_collapse_guard": (
                metrics["maximum_predicted_class_share"] <= 0.90
            ),
            "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
        }
    return result


def choose_candidate(evaluations: Sequence[CandidateEvaluation]) -> CandidateEvaluation:
    """Choose a non-collapsed candidate under the frozen comparison ordering."""
    by_name = {evaluation.name: evaluation for evaluation in evaluations}
    unknown = set(by_name) - set(CANDIDATE_ORDER)
    if unknown or len(by_name) != len(evaluations):
        raise ValueError(f"Candidate evaluations are duplicate or unknown: {sorted(unknown)}")
    eligible = [
        evaluation
        for evaluation in evaluations
        if evaluation.maximum_predicted_class_share <= 0.90
    ]
    if not eligible:
        raise ValueError("No candidate passes the class-collapse guard")
    order = {name: index for index, name in enumerate(CANDIDATE_ORDER)}
    return max(
        eligible,
        key=lambda item: (
            item.macro_f1_mean,
            item.f1_label0_mean,
            -item.macro_f1_std,
            -order[item.name],
        ),
    )
