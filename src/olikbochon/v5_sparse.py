"""Frozen official-only sparse text candidates for the V5 context-absent route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loading import validate_labeled_frame
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v4_diagnostics import probability_diagnostics
from .v4_preprocessing import official_context_is_present
from .v5_features import CONTEXT_ABSENT_FEATURES, V5FeatureExtractor
from .v5_lexical import (
    build_v5_folds,
    deterministic_substring_prediction,
    fold_feasibility_audit,
)
from .v5_normalization import normalize_v5


EXPERIMENT_NAME = "v5_null_sparse_baseline"
CANDIDATE_SET = "frozen"
CANDIDATES = (
    "candidate_e_char_tfidf",
    "candidate_f_word_tfidf",
    "candidate_g_char_word_union",
    "candidate_h_separate_fields",
    "candidate_i_sparse_lexical_union",
)
LOGISTIC_CONFIG = {
    "class_weight": "balanced",
    "C": 1.0,
    "max_iter": 3000,
    "solver": "liblinear",
    "random_state": 42,
}
UNLABELED_INFERENCE_COLUMNS = ("id", "context", "prompt_bn", "response_bn")


@dataclass(frozen=True)
class VectorizerSpec:
    analyzer: str
    ngram_range: tuple[int, int]
    lowercase: bool
    sublinear_tf: bool
    min_df: int
    max_features: int
    norm: str

    def build(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
            sublinear_tf=self.sublinear_tf,
            min_df=self.min_df,
            max_features=self.max_features,
            norm=self.norm,
        )


CHAR_SPEC = VectorizerSpec("char_wb", (3, 5), False, True, 2, 30000, "l2")
WORD_SPEC = VectorizerSpec("word", (1, 2), False, True, 2, 15000, "l2")
PROMPT_WORD_SPEC = VectorizerSpec("word", (1, 2), False, True, 2, 8000, "l2")
RESPONSE_CHAR_SPEC = VectorizerSpec("char_wb", (3, 5), False, True, 2, 20000, "l2")


def field_marked_text(prompt: Any, response: Any) -> str:
    """Return View 1 with ordinary markers and punctuation-preserving V5 normalization."""
    return f"[QUESTION] {normalize_v5(prompt)}\n[ANSWER] {normalize_v5(response)}"


def _require_absent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    if any(official_context_is_present(value) for value in validated["context"]):
        raise ValueError("Sparse null-route models accept context-absent rows only")
    return validated


def _require_absent_unlabeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != UNLABELED_INFERENCE_COLUMNS:
        raise ValueError(
            "Sparse unlabeled inference requires exact id/context/prompt/response columns"
        )
    if frame["id"].isna().any():
        raise ValueError("Sparse unlabeled inference IDs must not be missing")
    if frame["prompt_bn"].isna().any() or frame["response_bn"].isna().any():
        raise ValueError("Sparse unlabeled prompt and response values must not be missing")
    if any(official_context_is_present(value) for value in frame["context"]):
        raise ValueError("Sparse null-route models accept context-absent rows only")
    return frame.copy(deep=True).reset_index(drop=True)


def _finite_sparse(matrix: csr_matrix) -> None:
    if not issparse(matrix) or not np.isfinite(matrix.data).all():
        raise ValueError("Sparse feature matrix contains NaN or Inf")


class SparseNullModel:
    """One frozen sparse candidate with explicit train-only fitted state."""

    def __init__(self, candidate: str) -> None:
        if candidate not in CANDIDATES:
            raise ValueError(f"Unknown sparse candidate {candidate!r}")
        self.candidate = candidate
        self.vectorizers: dict[str, TfidfVectorizer] = {}
        self.numeric_extractor: V5FeatureExtractor | None = None
        self.numeric_scaler: StandardScaler | None = None
        self.classifier = LogisticRegression(**LOGISTIC_CONFIG)
        self.training_row_count: int | None = None

    def _channel_specs(self) -> tuple[tuple[str, VectorizerSpec, str], ...]:
        if self.candidate == CANDIDATES[0]:
            return (("combined_char", CHAR_SPEC, "combined"),)
        if self.candidate == CANDIDATES[1]:
            return (("combined_word", WORD_SPEC, "combined"),)
        if self.candidate in {CANDIDATES[2], CANDIDATES[4]}:
            return (
                ("combined_char", CHAR_SPEC, "combined"),
                ("combined_word", WORD_SPEC, "combined"),
            )
        return (
            ("prompt_word", PROMPT_WORD_SPEC, "prompt"),
            ("response_char", RESPONSE_CHAR_SPEC, "response"),
        )

    @staticmethod
    def _texts(frame: pd.DataFrame, view: str) -> list[str]:
        if view == "combined":
            return [
                field_marked_text(row.prompt_bn, row.response_bn)
                for row in frame.itertuples(index=False)
            ]
        column = "prompt_bn" if view == "prompt" else "response_bn"
        return [normalize_v5(value) for value in frame[column]]

    def _sparse_blocks(self, frame: pd.DataFrame, *, fit: bool) -> list[csr_matrix]:
        blocks: list[csr_matrix] = []
        for name, spec, view in self._channel_specs():
            texts = self._texts(frame, view)
            if fit:
                vectorizer = spec.build()
                matrix = vectorizer.fit_transform(texts).tocsr()
                self.vectorizers[name] = vectorizer
            else:
                if name not in self.vectorizers:
                    raise RuntimeError("Sparse model must be fitted before transform")
                matrix = self.vectorizers[name].transform(texts).tocsr()
            _finite_sparse(matrix)
            blocks.append(matrix)
        return blocks

    def _numeric_block(self, frame: pd.DataFrame, *, fit: bool) -> csr_matrix:
        indices = tuple(range(len(frame)))
        if fit:
            self.numeric_extractor = V5FeatureExtractor.fit(frame)
            numeric = self.numeric_extractor.matrix(
                frame, indices, context_present=False
            )
            self.numeric_scaler = StandardScaler()
            scaled = self.numeric_scaler.fit_transform(numeric)
        else:
            if self.numeric_extractor is None or self.numeric_scaler is None:
                raise RuntimeError("Numeric feature state must be fitted before transform")
            numeric = self.numeric_extractor.matrix(
                frame, indices, context_present=False
            )
            scaled = self.numeric_scaler.transform(numeric)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled numeric feature matrix contains NaN or Inf")
        return csr_matrix(scaled)

    def _matrix(self, frame: pd.DataFrame, *, fit: bool) -> csr_matrix:
        validated = _require_absent_frame(frame)
        return self._matrix_from_validated(validated, fit=fit)

    def _matrix_from_validated(
        self, validated: pd.DataFrame, *, fit: bool
    ) -> csr_matrix:
        blocks = self._sparse_blocks(validated, fit=fit)
        if self.candidate == CANDIDATES[4]:
            blocks.append(self._numeric_block(validated, fit=fit))
        matrix = hstack(blocks, format="csr")
        _finite_sparse(matrix)
        return matrix

    def fit(self, frame: pd.DataFrame) -> SparseNullModel:
        validated = _require_absent_frame(frame)
        labels = validated["label"].to_numpy(dtype=np.int64)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Sparse training fold must contain both labels")
        matrix = self._matrix(validated, fit=True)
        self.classifier.fit(matrix, labels)
        self.training_row_count = len(validated)
        return self

    def transform(self, frame: pd.DataFrame) -> csr_matrix:
        if self.training_row_count is None:
            raise RuntimeError("Sparse model must be fitted before transform")
        return self._matrix(frame, fit=False)

    def transform_unlabeled(self, frame: pd.DataFrame) -> csr_matrix:
        """Transform an exact unlabeled null-route frame without exposing any fit path."""
        if self.training_row_count is None:
            raise RuntimeError("Sparse model must be fitted before transform")
        validated = _require_absent_unlabeled_frame(frame)
        return self._matrix_from_validated(validated, fit=False)

    def predict_label1_probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.transform(frame)
        classes = list(self.classifier.classes_)
        probabilities = self.classifier.predict_proba(matrix)[:, classes.index(1)]
        if not np.isfinite(probabilities).all():
            raise ValueError("Sparse probabilities contain NaN or Inf")
        return probabilities.astype(np.float64)

    def predict_unlabeled_label1_probability(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.transform_unlabeled(frame)
        classes = list(self.classifier.classes_)
        probabilities = self.classifier.predict_proba(matrix)[:, classes.index(1)]
        if not np.isfinite(probabilities).all():
            raise ValueError("Sparse probabilities contain NaN or Inf")
        return probabilities.astype(np.float64)

    def fit_audit(self) -> dict[str, Any]:
        if self.training_row_count is None:
            raise RuntimeError("Sparse fit audit requires a fitted model")
        return {
            "candidate": self.candidate,
            "training_row_count": self.training_row_count,
            "vectorizer_fit_scope": "current_context_absent_training_fold_only",
            "vocabulary_sizes": {
                name: len(vectorizer.vocabulary_)
                for name, vectorizer in sorted(self.vectorizers.items())
            },
            "numeric_feature_count": (
                len(CONTEXT_ABSENT_FEATURES) if self.numeric_extractor is not None else 0
            ),
            "numeric_fit_scope": (
                "current_context_absent_training_fold_only"
                if self.numeric_extractor is not None
                else None
            ),
            "raw_vocabulary_persisted": False,
        }


def apply_routed_predictions(
    frame: pd.DataFrame,
    null_probabilities: Sequence[float],
    *,
    null_threshold: float,
) -> np.ndarray:
    """Use only Candidate A on present rows and sparse probabilities on absent rows."""
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    absent_count = int((~presence).sum())
    probabilities = np.asarray(null_probabilities, dtype=np.float64)
    if probabilities.shape != (absent_count,) or not np.isfinite(probabilities).all():
        raise ValueError("Null probabilities must be finite and aligned to absent rows")
    predictions = np.empty(len(frame), dtype=np.int64)
    predictions[~presence] = predictions_from_label1(probabilities, null_threshold)
    present_rows = frame.loc[presence]
    predictions[presence] = [
        deterministic_substring_prediction(row.context, row.response_bn)
        for row in present_rows.itertuples(index=False)
    ]
    return predictions


def _prediction_counts(predictions: np.ndarray) -> dict[str, int]:
    counts = np.bincount(predictions, minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def _metric_record(
    truth: np.ndarray, predictions: np.ndarray, presence: np.ndarray | None = None
) -> dict[str, Any]:
    result = {
        **classification_metrics(truth, predictions),
        "prediction_counts": _prediction_counts(predictions),
    }
    if presence is not None:
        result["route_metrics"] = subgroup_metrics(truth, predictions, presence)
    return result


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


def run_sparse_experiment(frame: pd.DataFrame) -> dict[str, Any]:
    """Run the frozen candidates and retain only aggregate/fold numeric records."""
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    folds = build_v5_folds(validated)
    feasibility = fold_feasibility_audit(validated, folds)
    truth = validated["label"].to_numpy(dtype=np.int64)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    pooled_null_probabilities: dict[str, list[np.ndarray]] = {
        candidate: [] for candidate in CANDIDATES
    }
    per_seed: dict[str, Any] = {}
    fold_records: list[dict[str, Any]] = []

    for seed in folds.seeds:
        seed_probabilities = {
            candidate: np.full(len(validated), np.nan, dtype=np.float64)
            for candidate in CANDIDATES
        }
        for split in (item for item in folds.folds if item.seed == seed):
            train = np.asarray(split.train_indices, dtype=np.int64)
            validation = np.asarray(split.validation_indices, dtype=np.int64)
            train_absent = train[~presence[train]]
            validation_absent = validation[~presence[validation]]
            null_train = validated.iloc[train_absent].reset_index(drop=True)
            null_validation = validated.iloc[validation_absent].reset_index(drop=True)
            fold_result: dict[str, Any] = {
                "seed": seed,
                "fold": split.fold,
                "group_overlap_count": 0,
                "candidates": {},
            }
            for candidate in CANDIDATES:
                model = SparseNullModel(candidate).fit(null_train)
                probabilities = model.predict_label1_probability(null_validation)
                seed_probabilities[candidate][validation_absent] = probabilities
                null_predictions = predictions_from_label1(probabilities, 0.5)
                diagnostics = probability_diagnostics(
                    truth[validation_absent], probabilities
                )
                routed_predictions = apply_routed_predictions(
                    validated.iloc[validation].reset_index(drop=True),
                    probabilities,
                    null_threshold=0.5,
                )
                fold_result["candidates"][candidate] = {
                    "fit_audit": model.fit_audit(),
                    "null_threshold_050": _metric_record(
                        truth[validation_absent], null_predictions
                    ),
                    "null_probability_diagnostics": diagnostics,
                    "routed_threshold_050": _metric_record(
                        truth[validation], routed_predictions, presence[validation]
                    ),
                }
            fold_records.append(fold_result)

        seed_result: dict[str, Any] = {}
        for candidate in CANDIDATES:
            probabilities = seed_probabilities[candidate][~presence]
            if not np.isfinite(probabilities).all():
                raise RuntimeError(
                    f"Incomplete null-route OOF coverage for {candidate}, seed {seed}"
                )
            pooled_null_probabilities[candidate].append(probabilities)
            diagnostics = probability_diagnostics(truth[~presence], probabilities)
            selected_threshold = float(diagnostics["best_frozen_grid"]["threshold"])
            routed_050 = apply_routed_predictions(
                validated, probabilities, null_threshold=0.5
            )
            routed_selected = apply_routed_predictions(
                validated, probabilities, null_threshold=selected_threshold
            )
            seed_result[candidate] = {
                "complete_null_oof_coverage": True,
                "null_threshold_050": _metric_record(
                    truth[~presence], predictions_from_label1(probabilities, 0.5)
                ),
                "null_probability_diagnostics": diagnostics,
                "routed_threshold_050": _metric_record(truth, routed_050, presence),
                "routed_selected_null_threshold": {
                    "null_threshold": selected_threshold,
                    **_metric_record(truth, routed_selected, presence),
                },
            }
        per_seed[str(seed)] = seed_result

    repeated_truth = np.tile(truth, len(folds.seeds))
    repeated_presence = np.tile(presence, len(folds.seeds))
    repeated_null_truth = np.tile(truth[~presence], len(folds.seeds))
    aggregate: dict[str, Any] = {}
    for candidate in CANDIDATES:
        probabilities = np.concatenate(pooled_null_probabilities[candidate])
        diagnostics = probability_diagnostics(repeated_null_truth, probabilities)
        selected_threshold = float(diagnostics["best_frozen_grid"]["threshold"])
        repeated_null_050 = predictions_from_label1(probabilities, 0.5)
        repeated_null_selected = predictions_from_label1(probabilities, selected_threshold)
        routed_050_parts: list[np.ndarray] = []
        routed_selected_parts: list[np.ndarray] = []
        null_050_seed_records: list[dict[str, Any]] = []
        null_selected_seed_records: list[dict[str, Any]] = []
        routed_050_seed_records: list[dict[str, Any]] = []
        selected_seed_records: list[dict[str, Any]] = []
        for seed_probabilities in pooled_null_probabilities[candidate]:
            null_050_seed_records.append(
                _metric_record(
                    truth[~presence],
                    predictions_from_label1(seed_probabilities, 0.5),
                )
            )
            null_selected_seed_records.append(
                _metric_record(
                    truth[~presence],
                    predictions_from_label1(seed_probabilities, selected_threshold),
                )
            )
            routed_050_predictions = apply_routed_predictions(
                validated, seed_probabilities, null_threshold=0.5
            )
            routed_050_parts.append(routed_050_predictions)
            routed_050_seed_records.append(
                _metric_record(truth, routed_050_predictions, presence)
            )
            selected_predictions = apply_routed_predictions(
                validated,
                seed_probabilities,
                null_threshold=selected_threshold,
            )
            routed_selected_parts.append(selected_predictions)
            selected_seed_records.append(_metric_record(truth, selected_predictions, presence))
        routed_050 = np.concatenate(routed_050_parts)
        routed_selected = np.concatenate(routed_selected_parts)
        selected_null_metrics = _metric_record(
            repeated_null_truth, repeated_null_selected
        )
        maximum_share = max(selected_null_metrics["prediction_counts"].values()) / len(
            repeated_null_selected
        )
        aggregate[candidate] = {
            "null_threshold_050": _metric_record(
                repeated_null_truth, repeated_null_050
            ),
            "null_threshold_050_seed_metric_distribution": _distribution(
                null_050_seed_records
            ),
            "null_probability_diagnostics": diagnostics,
            "null_selected_threshold": {
                "threshold": selected_threshold,
                **selected_null_metrics,
                "maximum_predicted_class_share": float(maximum_share),
                "passes_class_collapse_guard": bool(maximum_share <= 0.90),
                "seed_metric_distribution": _distribution(
                    null_selected_seed_records
                ),
            },
            "routed_threshold_050": _metric_record(
                repeated_truth, routed_050, repeated_presence
            ),
            "routed_threshold_050_seed_metric_distribution": _distribution(
                routed_050_seed_records
            ),
            "routed_selected_null_threshold": {
                "null_threshold": selected_threshold,
                **_metric_record(repeated_truth, routed_selected, repeated_presence),
                "seed_metric_distribution": _distribution(selected_seed_records),
            },
        }

    return {
        "experiment": EXPERIMENT_NAME,
        "candidate_set": CANDIDATE_SET,
        "candidates": list(CANDIDATES),
        "validation": feasibility,
        "fold_metrics": fold_records,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "row_level_values_persisted": False,
        "raw_vocabulary_persisted": False,
    }
