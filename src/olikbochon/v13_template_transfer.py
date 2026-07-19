"""Frozen V13 hierarchical public template and nearest-neighbour transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .metrics import classification_metrics


THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
NEIGHBOURS = 7


@dataclass(frozen=True)
class TransferOutput:
    predictions: np.ndarray
    probabilities: np.ndarray
    routes: np.ndarray
    maximum_similarity: np.ndarray


def _precision(truth: np.ndarray, prediction: np.ndarray, label: int) -> float | None:
    selected = prediction == label
    return float(np.mean(truth[selected] == label)) if selected.any() else None


def _rule_metrics(truth: np.ndarray, prediction: np.ndarray, covered: np.ndarray) -> dict[str, Any]:
    if not covered.any():
        return {"support": 0, "eligible_minimum_support": False}
    y = truth[covered]
    p = prediction[covered]
    return {
        "support": int(covered.sum()),
        **classification_metrics(y, p),
        "precision_label0": _precision(y, p, 0),
        "precision_label1": _precision(y, p, 1),
        "mixed_label_conflict_rate": 0.0,
        "eligible_minimum_support": int(covered.sum()) >= 30,
    }


class FrozenTemplateTransfer:
    def __init__(self, train: pd.DataFrame) -> None:
        self.train = train.reset_index(drop=True)
        self.labels = self.train.label.to_numpy(dtype=np.int64)
        self.character = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1,
            max_features=120_000, norm="l2"
        )
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), sublinear_tf=True, min_df=1,
            max_features=60_000, norm="l2"
        )
        self.character_matrix = self.character.fit_transform(self.train.pair_key).tocsr()
        self.word_matrix = self.word.fit_transform(self.train.pair_key).tocsr()
        self.pair_labels = self._label_map("pair_key")
        skeleton_groups = self.train.groupby("skeleton_key").label.agg(["size", "mean"])
        self.skeleton_labels = {
            key: int(row["mean"] >= 0.5)
            for key, row in skeleton_groups.iterrows()
            if row["size"] >= 3 and max(row["mean"], 1.0 - row["mean"]) >= 0.98
        }
        self.exact_enabled = False
        self.skeleton_enabled = False
        self.nn_threshold: float | None = None
        self.calibration: dict[str, Any] = {}

    def _label_map(self, key: str) -> dict[str, int]:
        grouped = self.train.groupby(key).label.agg(["mean", "nunique"])
        return {value: int(row["mean"] >= 0.5) for value, row in grouped.iterrows() if row["nunique"] == 1}

    def _similarities(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            0.70 * (self.character.transform(frame.pair_key) @ self.character_matrix.T).toarray()
            + 0.30 * (self.word.transform(frame.pair_key) @ self.word_matrix.T).toarray()
        )

    def _nn(self, similarities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        top = np.argpartition(-similarities, kth=NEIGHBOURS - 1, axis=1)[:, :NEIGHBOURS]
        top_scores = np.take_along_axis(similarities, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        top_scores = np.take_along_axis(top_scores, order, axis=1)
        weights = np.power(np.clip(top_scores, 0.0, None), 4)
        denominator = weights.sum(axis=1)
        probability = np.divide(
            (weights * self.labels[top]).sum(axis=1), denominator,
            out=np.full(len(similarities), 0.5), where=denominator > 0,
        )
        return probability, (probability >= 0.5).astype(np.int64), top_scores[:, 0]

    def calibrate(self, validation: pd.DataFrame) -> dict[str, Any]:
        truth = validation.label.to_numpy(dtype=np.int64)
        exact_prediction = np.asarray([self.pair_labels.get(key, -1) for key in validation.pair_key])
        exact_covered = exact_prediction >= 0
        exact_metrics = _rule_metrics(truth, exact_prediction, exact_covered)
        self.exact_enabled = (
            exact_metrics.get("support", 0) >= 30
            and exact_metrics.get("accuracy", 0.0) >= 0.95
        )
        skeleton_prediction = np.asarray([
            self.skeleton_labels.get(key, -1) for key in validation.skeleton_key
        ])
        skeleton_covered = skeleton_prediction >= 0
        skeleton_metrics = _rule_metrics(truth, skeleton_prediction, skeleton_covered)
        precision_values = [
            value for value in (
                skeleton_metrics.get("precision_label0"), skeleton_metrics.get("precision_label1")
            ) if value is not None
        ]
        self.skeleton_enabled = (
            skeleton_metrics.get("support", 0) >= 30
            and bool(precision_values)
            and min(precision_values) >= 0.95
        )
        similarities = self._similarities(validation)
        probability, prediction, maximum = self._nn(similarities)
        threshold_records = []
        selected = None
        for threshold in THRESHOLDS:
            covered = maximum >= threshold
            metrics = _rule_metrics(truth, prediction, covered)
            record = {"threshold": threshold, **metrics}
            threshold_records.append(record)
            if (
                selected is None
                and metrics.get("support", 0) >= 50
                and metrics.get("accuracy", 0.0) >= 0.95
                and metrics.get("macro_f1", 0.0) >= 0.90
            ):
                selected = threshold
        self.nn_threshold = selected
        self.calibration = {
            "exact": exact_metrics,
            "exact_enabled": self.exact_enabled,
            "skeleton": skeleton_metrics,
            "skeleton_enabled": self.skeleton_enabled,
            "nearest_neighbour_all": {
                **classification_metrics(truth, prediction),
                "precision_label0": _precision(truth, prediction, 0),
                "precision_label1": _precision(truth, prediction, 1),
            },
            "nearest_neighbour_thresholds": threshold_records,
            "selected_nn_threshold": selected,
            "maximum_similarity_quantiles": {
                str(q): float(np.quantile(maximum, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
        }
        return self.calibration

    def predict(self, frame: pd.DataFrame) -> TransferOutput:
        similarities = self._similarities(frame)
        probability, prediction, maximum = self._nn(similarities)
        routes = np.full(len(frame), "nearest_neighbour", dtype=object)
        for position, row in enumerate(frame.itertuples(index=False)):
            if self.exact_enabled and row.pair_key in self.pair_labels:
                prediction[position] = self.pair_labels[row.pair_key]
                probability[position] = float(prediction[position])
                routes[position] = "exact_pair"
            elif self.skeleton_enabled and row.skeleton_key in self.skeleton_labels:
                prediction[position] = self.skeleton_labels[row.skeleton_key]
                probability[position] = float(prediction[position])
                routes[position] = "skeleton"
            elif self.nn_threshold is not None and maximum[position] >= self.nn_threshold:
                routes[position] = "high_confidence_nn"
        return TransferOutput(prediction, probability, routes, maximum)

    def overlap_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        output = self.predict(frame)
        counts = pd.Series(output.routes).value_counts()
        return {
            "rows": len(frame),
            "route_counts": {str(key): int(value) for key, value in counts.items()},
            "transfer_eligible_count": int(np.isin(output.routes, ("exact_pair", "skeleton", "high_confidence_nn")).sum()),
            "transfer_eligible_fraction": float(np.isin(output.routes, ("exact_pair", "skeleton", "high_confidence_nn")).mean()),
            "similarity_quantiles": {str(q): float(np.quantile(output.maximum_similarity, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
            "coverage_by_threshold": {str(value): int((output.maximum_similarity >= value).sum()) for value in THRESHOLDS},
        }
