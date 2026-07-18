"""Constrained nested-grouped runner for frozen V7 retrieval candidates M, N, and P."""

from __future__ import annotations

import argparse
import ctypes
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loading import validate_labeled_frame
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v4_preprocessing import official_context_is_present
from .v4_runner import load_official_training_frame
from .v4_validation import FOLD_COUNT, THRESHOLD_GRID, VALIDATION_SEEDS
from .v5_features import CONTEXT_ABSENT_FEATURES, V5FeatureExtractor
from .v5_lexical import (
    build_v5_folds,
    deterministic_substring_prediction,
    fold_feasibility_audit,
)
from .v7_corpus import MINIMUM_ARTICLE_CHARACTERS, load_corpus, require_corpus_path
from .v7_features import (
    TOP5_FEATURES,
    V4A_FEATURES,
    feature_vector,
    top5_features,
    v4a_features,
)
from .v7_retrieval import CharacterTfidfRetriever, RetrievedPassage


EXPERIMENT_NAME = "v7_null_retrieval_verification"
MODE = "null-retrieval-verification"
CANDIDATE_SET = "frozen"
CANDIDATE_M = "candidate_m_v4a_top1_reproduction"
CANDIDATE_N = "candidate_n_top5_aggregate"
CANDIDATE_P = "candidate_p_retrieval_feature_stacker"
CANDIDATES = (CANDIDATE_M, CANDIDATE_N, CANDIDATE_P)
OUTPUT_NAME = "null_retrieval_verification"
RETRIEVAL_CUTOFF_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
INNER_FOLDS = 3
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169
NULL_CHAMPION = 0.541763
ROUTED_CHAMPION = 0.692051
EXPECTED_LOGICAL_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
EXPECTED_UNIQUE_CHUNKS = 301
EXPECTED_USABLE_ARTICLES = 62_153
MAXIMUM_PROCESS_RAM_BYTES = 24 * 1024**3
CLASSIFIER_CONFIG = {
    "class_weight": "balanced",
    "C": 1.0,
    "max_iter": 3000,
    "solver": "liblinear",
    "random_state": 42,
}


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory_bytes() -> dict[str, int]:
    """Read current and peak process working set without an external dependency."""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_current_process.restype = ctypes.c_void_p
    get_process_memory.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    get_process_memory.restype = ctypes.c_int
    handle = get_current_process()
    if not get_process_memory(handle, ctypes.byref(counters), counters.cb):
        raise OSError("Unable to query V7 process memory")
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def require_memory_safe() -> dict[str, int]:
    memory = process_memory_bytes()
    if max(memory.values()) > MAXIMUM_PROCESS_RAM_BYTES:
        raise MemoryError("V7 process RAM exceeded the frozen 24 GB safety limit")
    return memory


@dataclass(frozen=True)
class EvidenceRows:
    intro_top1: tuple[tuple[RetrievedPassage, ...], ...]
    centered_top5: tuple[tuple[RetrievedPassage, ...], ...]

    def subset(self, positions: Sequence[int]) -> EvidenceRows:
        selected = tuple(int(position) for position in positions)
        return EvidenceRows(
            tuple(self.intro_top1[position] for position in selected),
            tuple(self.centered_top5[position] for position in selected),
        )


def _require_null_frame(frame: pd.DataFrame, *, require_both_labels: bool) -> pd.DataFrame:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    if any(official_context_is_present(value) for value in validated["context"]):
        raise ValueError("V7 classifiers accept context-absent official rows only")
    if require_both_labels and set(validated["label"].unique()) != {0, 1}:
        raise ValueError("V7 train/validation partitions must contain both labels")
    return validated


def _build_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(**CLASSIFIER_CONFIG)),
        ]
    )


class RetrievalCandidateModel:
    """One low-capacity candidate with train-fold-only lexical fitted state."""

    def __init__(self, candidate: str, cutoff: float) -> None:
        if candidate not in CANDIDATES:
            raise ValueError(f"Unknown V7 candidate {candidate!r}")
        if cutoff not in RETRIEVAL_CUTOFF_GRID:
            raise ValueError("Retrieval cutoff is outside the frozen grid")
        self.candidate = candidate
        self.cutoff = cutoff
        self.classifier = _build_classifier()
        self.lexical_extractor: V5FeatureExtractor | None = None
        self.training_rows: int | None = None

    def _evidence_matrix(
        self,
        frame: pd.DataFrame,
        evidence: EvidenceRows,
        *,
        fit: bool,
    ) -> np.ndarray:
        validated = _require_null_frame(frame, require_both_labels=fit)
        if len(validated) != len(evidence.intro_top1) or len(validated) != len(
            evidence.centered_top5
        ):
            raise ValueError("V7 evidence and official rows are not aligned")
        rows: list[np.ndarray] = []
        for position, row in enumerate(validated.itertuples(index=False)):
            if self.candidate == CANDIDATE_M:
                record = v4a_features(
                    row.response_bn,
                    evidence.intro_top1[position],
                    cutoff=self.cutoff,
                )
                rows.append(feature_vector(record, V4A_FEATURES))
                continue
            top_record = top5_features(
                row.response_bn,
                evidence.centered_top5[position],
                cutoff=self.cutoff,
            )
            blocks = [feature_vector(top_record, TOP5_FEATURES)]
            if self.candidate == CANDIDATE_P:
                v4a_record = v4a_features(
                    row.response_bn,
                    evidence.centered_top5[position],
                    cutoff=self.cutoff,
                )
                blocks.insert(0, feature_vector(v4a_record, V4A_FEATURES))
            rows.append(np.concatenate(blocks))
        matrix = np.asarray(rows, dtype=np.float64)
        if self.candidate == CANDIDATE_P:
            if fit:
                self.lexical_extractor = V5FeatureExtractor.fit(validated)
            if self.lexical_extractor is None:
                raise RuntimeError("Candidate P lexical features must be fitted on training rows")
            lexical = self.lexical_extractor.matrix(
                validated,
                tuple(range(len(validated))),
                context_present=False,
            )
            matrix = np.hstack((matrix, lexical))
        if not np.isfinite(matrix).all():
            raise ValueError("V7 candidate matrix contains NaN or Inf")
        return matrix

    def fit(self, frame: pd.DataFrame, evidence: EvidenceRows) -> RetrievalCandidateModel:
        validated = _require_null_frame(frame, require_both_labels=True)
        matrix = self._evidence_matrix(validated, evidence, fit=True)
        self.classifier.fit(matrix, validated["label"].to_numpy(dtype=np.int64))
        self.training_rows = len(validated)
        return self

    def predict_probability(self, frame: pd.DataFrame, evidence: EvidenceRows) -> np.ndarray:
        if self.training_rows is None:
            raise RuntimeError("V7 candidate must be fitted before prediction")
        validated = _require_null_frame(frame, require_both_labels=False)
        matrix = self._evidence_matrix(validated, evidence, fit=False)
        logistic = self.classifier.named_steps["logistic"]
        probabilities = self.classifier.predict_proba(matrix)[:, list(logistic.classes_).index(1)]
        if not np.isfinite(probabilities).all():
            raise ValueError("V7 probabilities contain NaN or Inf")
        return probabilities.astype(np.float64)

    def fit_audit(self) -> dict[str, Any]:
        if self.training_rows is None:
            raise RuntimeError("V7 fit audit requires a fitted classifier")
        base_features = (
            V4A_FEATURES
            if self.candidate == CANDIDATE_M
            else TOP5_FEATURES
            if self.candidate == CANDIDATE_N
            else V4A_FEATURES + TOP5_FEATURES + CONTEXT_ABSENT_FEATURES
        )
        return {
            "candidate": self.candidate,
            "training_rows": self.training_rows,
            "retrieval_cutoff": self.cutoff,
            "feature_count": len(base_features),
            "lexical_fit_scope": (
                "current_null_route_training_partition_only"
                if self.candidate == CANDIDATE_P
                else None
            ),
            "raw_text_persisted": False,
            "row_level_evidence_persisted": False,
        }


def build_inner_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    outer_seed: int,
    outer_fold: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if set(np.unique(labels)) != {0, 1} or labels.shape != groups.shape:
        raise ValueError("Inner grouped selection requires aligned binary labels")
    splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=outer_seed * 100 + outer_fold,
    )
    rows = np.arange(len(labels), dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    assignments = np.zeros(len(labels), dtype=np.int64)
    for train, validation in splitter.split(rows, labels, groups):
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("V7 inner grouped split leaked a strengthened group")
        if set(np.unique(labels[train])) != {0, 1} or set(np.unique(labels[validation])) != {
            0,
            1,
        }:
            raise RuntimeError("V7 inner split lacks one official label")
        assignments[validation] += 1
        splits.append((train.astype(np.int64), validation.astype(np.int64)))
    if not np.all(assignments == 1):
        raise RuntimeError("V7 inner selection did not cover each training row once")
    return tuple(splits)


def _class_share(predictions: np.ndarray) -> float:
    return float(np.bincount(predictions, minlength=2).max() / len(predictions))


def select_inner_parameters(
    candidate: str,
    frame: pd.DataFrame,
    evidence: EvidenceRows,
    groups: np.ndarray,
    *,
    outer_seed: int,
    outer_fold: int,
) -> dict[str, Any]:
    """Select cutoff and classifier threshold using inner OOF predictions only."""
    validated = _require_null_frame(frame, require_both_labels=True)
    labels = validated["label"].to_numpy(dtype=np.int64)
    splits = build_inner_splits(
        labels,
        groups,
        outer_seed=outer_seed,
        outer_fold=outer_fold,
    )
    ranked: list[tuple[tuple[float, ...], float, float, dict[str, Any]]] = []
    for cutoff in RETRIEVAL_CUTOFF_GRID:
        probabilities = np.full(len(validated), np.nan, dtype=np.float64)
        for train, validation in splits:
            model = RetrievalCandidateModel(candidate, cutoff).fit(
                validated.iloc[train].reset_index(drop=True), evidence.subset(train)
            )
            probabilities[validation] = model.predict_probability(
                validated.iloc[validation].reset_index(drop=True),
                evidence.subset(validation),
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("V7 inner OOF probabilities are incomplete")
        for threshold in THRESHOLD_GRID:
            predictions = predictions_from_label1(probabilities, threshold)
            share = _class_share(predictions)
            if share > 0.90:
                continue
            metrics = classification_metrics(labels, predictions)
            rank = (
                float(metrics["macro_f1"]),
                float(metrics["f1_label0"]),
                -abs(threshold - 0.50),
                -abs(cutoff - 0.25),
                -threshold,
                -cutoff,
            )
            ranked.append((rank, cutoff, threshold, {**metrics, "class_share": share}))
    if not ranked:
        raise RuntimeError("Every V7 inner parameter pair violates the collapse guard")
    _rank, cutoff, threshold, metrics = max(ranked, key=lambda item: item[0])
    return {
        "retrieval_cutoff": cutoff,
        "classifier_threshold": threshold,
        "inner_metrics": metrics,
        "selection_scope": "current_outer_training_partition_inner_grouped_oof_only",
        "outer_validation_rows_used_for_selection": 0,
    }


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


def _distribution(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([record["macro_f1"] for record in records], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _retrieval_acceptance_mask(
    evidence: EvidenceRows, candidate: str, cutoff: float
) -> np.ndarray:
    return _accepted_passage_counts(evidence, candidate, cutoff) > 0


def _accepted_passage_counts(
    evidence: EvidenceRows, candidate: str, cutoff: float
) -> np.ndarray:
    rows = evidence.intro_top1 if candidate == CANDIDATE_M else evidence.centered_top5
    return np.asarray(
        [sum(passage.score >= cutoff for passage in passages) for passages in rows],
        dtype=np.int64,
    )


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("V7 numeric summary requires finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _support_feature_records(
    frame: pd.DataFrame,
    evidence: EvidenceRows,
    candidate: str,
    cutoff: float,
) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        if candidate == CANDIDATE_M:
            records.append(
                v4a_features(row.response_bn, evidence.intro_top1[position], cutoff=cutoff)
            )
            continue
        record = top5_features(
            row.response_bn,
            evidence.centered_top5[position],
            cutoff=cutoff,
        )
        if candidate == CANDIDATE_P:
            record = {
                **v4a_features(
                    row.response_bn,
                    evidence.centered_top5[position],
                    cutoff=cutoff,
                ),
                **record,
            }
        records.append(record)
    return records


def run_experiment(
    frame: pd.DataFrame,
    retriever: CharacterTfidfRetriever,
) -> dict[str, Any]:
    """Run the frozen nested outer evaluation without persisting row-level evidence."""
    started = perf_counter()
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V7 corrected route totals differ from 130/169")
    folds = build_v5_folds(validated)
    feasibility = fold_feasibility_audit(validated, folds)
    labels = validated["label"].to_numpy(dtype=np.int64)
    group_ids = np.asarray(folds.audit.group_ids, dtype=object)
    absent_indices = np.flatnonzero(~presence)
    absent_positions = {int(index): position for position, index in enumerate(absent_indices)}
    null_frame = validated.iloc[absent_indices].reset_index(drop=True)
    queries = [str(value) for value in null_frame["prompt_bn"]]
    evidence = EvidenceRows(
        retriever.retrieve_queries(queries, top_k=1, snippet_method="intro"),
        retriever.retrieve_queries(queries, top_k=5, snippet_method="query_centered"),
    )
    memory_after_retrieval = require_memory_safe()
    top1_scores = [float(passages[0].score) for passages in evidence.centered_top5]
    top1_top2_gaps = [
        float(passages[0].score - passages[1].score)
        for passages in evidence.centered_top5
    ]
    present_predictions = np.asarray(
        [
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in validated.loc[presence].itertuples(index=False)
        ],
        dtype=np.int64,
    )
    candidate_results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_started = perf_counter()
        seed_null_predictions: dict[int, np.ndarray] = {
            seed: np.full(len(absent_indices), -1, dtype=np.int64) for seed in VALIDATION_SEEDS
        }
        seed_null_050: dict[int, np.ndarray] = {
            seed: np.full(len(absent_indices), -1, dtype=np.int64) for seed in VALIDATION_SEEDS
        }
        seed_probabilities: dict[int, np.ndarray] = {
            seed: np.full(len(absent_indices), np.nan, dtype=np.float64)
            for seed in VALIDATION_SEEDS
        }
        seed_accepted: dict[int, np.ndarray] = {
            seed: np.zeros(len(absent_indices), dtype=bool) for seed in VALIDATION_SEEDS
        }
        seed_accepted_counts: dict[int, np.ndarray] = {
            seed: np.full(len(absent_indices), -1, dtype=np.int64)
            for seed in VALIDATION_SEEDS
        }
        support_values: dict[int, dict[str, list[float]]] = {0: {}, 1: {}}
        fold_records: list[dict[str, Any]] = []
        for outer in folds.folds:
            train = np.asarray(outer.train_indices, dtype=np.int64)
            validation = np.asarray(outer.validation_indices, dtype=np.int64)
            if set(group_ids[train]) & set(group_ids[validation]):
                raise RuntimeError("V7 outer grouped split leaked a strengthened group")
            train_absent = train[~presence[train]]
            validation_absent = validation[~presence[validation]]
            train_positions = np.asarray(
                [absent_positions[int(index)] for index in train_absent], dtype=np.int64
            )
            validation_positions = np.asarray(
                [absent_positions[int(index)] for index in validation_absent], dtype=np.int64
            )
            train_frame = validated.iloc[train_absent].reset_index(drop=True)
            validation_frame = validated.iloc[validation_absent].reset_index(drop=True)
            train_evidence = evidence.subset(train_positions)
            validation_evidence = evidence.subset(validation_positions)
            selection = select_inner_parameters(
                candidate,
                train_frame,
                train_evidence,
                group_ids[train_absent],
                outer_seed=outer.seed,
                outer_fold=outer.fold,
            )
            model = RetrievalCandidateModel(
                candidate, float(selection["retrieval_cutoff"])
            ).fit(train_frame, train_evidence)
            probabilities = model.predict_probability(validation_frame, validation_evidence)
            selected_predictions = predictions_from_label1(
                probabilities, float(selection["classifier_threshold"])
            )
            primary_predictions = predictions_from_label1(probabilities, 0.50)
            seed_probabilities[outer.seed][validation_positions] = probabilities
            seed_null_predictions[outer.seed][validation_positions] = selected_predictions
            seed_null_050[outer.seed][validation_positions] = primary_predictions
            acceptance_mask = _retrieval_acceptance_mask(
                validation_evidence,
                candidate,
                float(selection["retrieval_cutoff"]),
            )
            accepted_counts = _accepted_passage_counts(
                validation_evidence,
                candidate,
                float(selection["retrieval_cutoff"]),
            )
            seed_accepted[outer.seed][validation_positions] = acceptance_mask
            seed_accepted_counts[outer.seed][validation_positions] = accepted_counts
            support_records = _support_feature_records(
                validation_frame,
                validation_evidence,
                candidate,
                float(selection["retrieval_cutoff"]),
            )
            validation_labels = labels[validation_absent]
            for label, record in zip(validation_labels, support_records, strict=True):
                for name, value in record.items():
                    support_values[int(label)].setdefault(name, []).append(float(value))
            fold_memory = require_memory_safe()
            fold_records.append(
                {
                    "candidate": candidate,
                    "seed": outer.seed,
                    "fold": outer.fold,
                    "train_rows": int(train_absent.size),
                    "validation_rows": int(validation_absent.size),
                    "train_groups": int(len(set(group_ids[train_absent]))),
                    "validation_groups": int(len(set(group_ids[validation_absent]))),
                    "group_overlap_count": 0,
                    "selection": selection,
                    "fit_audit": model.fit_audit(),
                    "retrieval_accepted": int(acceptance_mask.sum()),
                    "retrieval_rejected": int((~acceptance_mask).sum()),
                    "accepted_passage_count_distribution": dict(
                        Counter(str(int(value)) for value in accepted_counts)
                    ),
                    "process_memory": fold_memory,
                    "null_threshold_050": _metric_record(
                        labels[validation_absent], primary_predictions
                    ),
                    "null_selected": _metric_record(
                        labels[validation_absent], selected_predictions
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "candidate": candidate,
                        "seed": outer.seed,
                        "fold": outer.fold,
                        "status": "complete",
                        "null_selected_macro_f1": fold_records[-1]["null_selected"][
                            "macro_f1"
                        ],
                        "peak_process_ram_bytes": fold_memory[
                            "peak_working_set_bytes"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        seed_null_records: list[dict[str, Any]] = []
        seed_routed_records: list[dict[str, Any]] = []
        all_probabilities: list[np.ndarray] = []
        all_null_predictions: list[np.ndarray] = []
        all_null_050: list[np.ndarray] = []
        all_routed: list[np.ndarray] = []
        all_routed_050: list[np.ndarray] = []
        all_accepted: list[np.ndarray] = []
        all_accepted_counts: list[np.ndarray] = []
        null_truth = labels[~presence]
        for seed in VALIDATION_SEEDS:
            if (
                np.any(seed_null_predictions[seed] < 0)
                or np.any(seed_null_050[seed] < 0)
                or not np.isfinite(seed_probabilities[seed]).all()
            ):
                raise RuntimeError("V7 outer OOF coverage is incomplete")
            null_predictions = seed_null_predictions[seed]
            null_primary = seed_null_050[seed]
            routed = np.empty(len(validated), dtype=np.int64)
            routed_primary = np.empty(len(validated), dtype=np.int64)
            routed[presence] = present_predictions
            routed_primary[presence] = present_predictions
            routed[~presence] = null_predictions
            routed_primary[~presence] = null_primary
            seed_null_records.append(_metric_record(null_truth, null_predictions))
            seed_routed_records.append(_metric_record(labels, routed, presence))
            all_probabilities.append(seed_probabilities[seed])
            all_null_predictions.append(null_predictions)
            all_null_050.append(null_primary)
            all_routed.append(routed)
            all_routed_050.append(routed_primary)
            all_accepted.append(seed_accepted[seed])
            if np.any(seed_accepted_counts[seed] < 0):
                raise RuntimeError("V7 accepted-passage coverage is incomplete")
            all_accepted_counts.append(seed_accepted_counts[seed])
        repeated_null_truth = np.tile(null_truth, len(VALIDATION_SEEDS))
        repeated_truth = np.tile(labels, len(VALIDATION_SEEDS))
        repeated_presence = np.tile(presence, len(VALIDATION_SEEDS))
        probability_values = np.concatenate(all_probabilities)
        selected_values = np.concatenate(all_null_predictions)
        primary_values = np.concatenate(all_null_050)
        selected_share = _class_share(selected_values)
        repeated_acceptance = np.concatenate(all_accepted)
        repeated_accepted_counts = np.concatenate(all_accepted_counts)
        outcome_metrics: dict[str, Any] = {}
        for outcome, mask in (
            ("accepted", repeated_acceptance),
            ("rejected_or_empty", ~repeated_acceptance),
        ):
            outcome_metrics[outcome] = (
                {
                    "row_count": int(mask.sum()),
                    **_metric_record(repeated_null_truth[mask], selected_values[mask]),
                }
                if mask.any()
                else {"row_count": 0, "metrics_available": False}
            )
        candidate_memory = require_memory_safe()
        candidate_results[candidate] = {
            "folds": fold_records,
            "null_threshold_050": _metric_record(repeated_null_truth, primary_values),
            "null_selected_nested": {
                **_metric_record(repeated_null_truth, selected_values),
                "brier_score": float(
                    np.mean(np.square(probability_values - repeated_null_truth))
                ),
                "probability_mean": float(probability_values.mean()),
                "probability_std": float(probability_values.std(ddof=0)),
                "seed_macro_f1_distribution": _distribution(seed_null_records),
                "maximum_predicted_class_share": selected_share,
                "passes_class_collapse_guard": selected_share <= 0.90,
                "retrieval_cutoff_distribution": dict(
                    Counter(str(record["selection"]["retrieval_cutoff"]) for record in fold_records)
                ),
                "classifier_threshold_distribution": dict(
                    Counter(
                        str(record["selection"]["classifier_threshold"])
                        for record in fold_records
                    )
                ),
                "retrieval_accepted": sum(record["retrieval_accepted"] for record in fold_records),
                "retrieval_rejected": sum(record["retrieval_rejected"] for record in fold_records),
                "retrieval_coverage": float(repeated_acceptance.mean()),
                "no_accepted_evidence_percentage": float(
                    100.0 * np.mean(repeated_accepted_counts == 0)
                ),
                "accepted_passage_count_distribution": dict(
                    Counter(str(int(value)) for value in repeated_accepted_counts)
                ),
                "retrieval_outcome_metrics": outcome_metrics,
                "support_feature_summaries_by_true_label": {
                    str(label): {
                        name: _numeric_summary(values)
                        for name, values in sorted(feature_values.items())
                    }
                    for label, feature_values in support_values.items()
                },
            },
            "routed_threshold_050": _metric_record(
                repeated_truth, np.concatenate(all_routed_050), repeated_presence
            ),
            "routed_selected_nested": {
                **_metric_record(
                    repeated_truth, np.concatenate(all_routed), repeated_presence
                ),
                "seed_macro_f1_distribution": _distribution(seed_routed_records),
            },
            "runtime_seconds": perf_counter() - candidate_started,
            "peak_process_ram_bytes": candidate_memory["peak_working_set_bytes"],
        }
    return {
        "experiment": EXPERIMENT_NAME,
        "candidates": list(CANDIDATES),
        "validation": feasibility,
        "candidate_results": candidate_results,
        "retrieval_diagnostics": {
            "top1_score": _numeric_summary(top1_scores),
            "top1_top2_score_gap": _numeric_summary(top1_top2_gaps),
            "process_memory_after_retrieval": memory_after_retrieval,
        },
        "runtime_seconds": perf_counter() - started,
        "row_level_probabilities_persisted": False,
        "row_level_evidence_persisted": False,
        "raw_article_text_persisted": False,
    }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_output_path(root: Path, output: Path) -> Path:
    root = Path(root).resolve()
    rules = {
        line.strip()
        for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    if "artifacts/v7/" not in rules:
        raise ValueError("artifacts/v7/ must be explicitly ignored")
    if not output.is_absolute():
        raise ValueError("V7 output must be an absolute path")
    expected = (root / "artifacts" / "v7" / OUTPUT_NAME).resolve()
    resolved = output.resolve()
    if resolved != expected:
        raise ValueError(f"V7 output must be artifacts/v7/{OUTPUT_NAME}")
    if resolved.exists():
        raise ValueError("V7 output must not already exist")
    return resolved


def require_approved_corpus_path(root: Path, corpus: Path) -> Path:
    resolved = require_corpus_path(corpus)
    expected = (Path(root).resolve() / "data" / "retrieval" / "bnwiki").resolve()
    if resolved != expected:
        raise ValueError("V7 corpus must be the approved data/retrieval/bnwiki directory")
    return resolved


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v7_runner",
        description="Run the frozen official-only V7 null-route retrieval experiment.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--candidate-set", required=True, choices=(CANDIDATE_SET,))
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_arguments(args: argparse.Namespace, root: Path) -> tuple[Path, Path]:
    if args.mode != MODE or args.candidate_set != CANDIDATE_SET:
        raise ValueError("V7 requires the frozen M/N/P candidate set")
    if tuple(args.seeds) != VALIDATION_SEEDS or args.folds != FOLD_COUNT:
        raise ValueError("V7 requires seeds 17 29 43 and five folds")
    return (
        require_approved_corpus_path(root, args.corpus_path),
        require_output_path(root, args.output_dir),
    )


def frozen_config() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "candidates": list(CANDIDATES),
        "candidate_o_nli": "unimplemented_pending_separate_model_approval",
        "corpus": {
            "source": "abyaadrafid/bnwiki",
            "minimum_article_characters": MINIMUM_ARTICLE_CHARACTERS,
            "fit_scope": "unlabeled_corpus_only_no_official_queries",
        },
        "retrieval_cutoff_grid": list(RETRIEVAL_CUTOFF_GRID),
        "classifier_threshold_grid": list(THRESHOLD_GRID),
        "outer_validation": {"seeds": list(VALIDATION_SEEDS), "folds": FOLD_COUNT},
        "inner_validation_folds": INNER_FOLDS,
        "classifier": CLASSIFIER_CONFIG,
        "ablations": [
            "frozen_null_lexical_baseline_without_retrieval",
            CANDIDATE_M,
            CANDIDATE_N,
            CANDIDATE_P,
        ],
        "primary_snippets": {
            CANDIDATE_M: "intro_first_800_characters",
            CANDIDATE_N: "query_centered_800_character_window",
            CANDIDATE_P: "query_centered_800_character_window",
        },
        "artifact_contract": "aggregate numeric metadata only; no raw article/query/evidence or row-level probabilities",
    }


def main(argv: list[str] | None = None) -> int:
    overall_started = perf_counter()
    args = build_cli_parser().parse_args(argv)
    root = repository_root()
    corpus_path, output = validate_arguments(args, root)
    corpus_started = perf_counter()
    articles, manifest = load_corpus(corpus_path)
    corpus_load_seconds = perf_counter() - corpus_started
    if manifest.logical_content_manifest_sha256 != EXPECTED_LOGICAL_MANIFEST:
        raise RuntimeError("V7 corpus logical manifest differs from the authenticated corpus")
    if manifest.unique_chunk_count != EXPECTED_UNIQUE_CHUNKS:
        raise RuntimeError("V7 corpus unique chunk count differs from the authenticated corpus")
    if manifest.usable_article_count != EXPECTED_USABLE_ARTICLES:
        raise RuntimeError("V7 corpus usable article count differs from the authenticated corpus")
    memory_after_corpus = require_memory_safe()
    print(
        json.dumps(
            {
                "logical_manifest_sha256": manifest.logical_content_manifest_sha256,
                "unique_chunks": manifest.unique_chunk_count,
                "usable_articles": manifest.usable_article_count,
                "status": "corpus_authenticated",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    retriever = CharacterTfidfRetriever(articles)
    memory_after_index = require_memory_safe()
    print(
        json.dumps(
            {
                **retriever.fit_audit,
                "peak_process_ram_bytes": memory_after_index["peak_working_set_bytes"],
                "status": "retrieval_index_built",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frame = load_official_training_frame(root)
    result = run_experiment(frame, retriever)
    final_memory = require_memory_safe()
    result["corpus_manifest"] = manifest.to_dict()
    result["retrieval_index"] = retriever.fit_audit
    result["execution_audit"] = {
        "corpus_load_seconds": corpus_load_seconds,
        "memory_after_corpus": memory_after_corpus,
        "memory_after_index": memory_after_index,
        "final_memory": final_memory,
        "total_runtime_seconds": perf_counter() - overall_started,
        "maximum_process_ram_bytes": MAXIMUM_PROCESS_RAM_BYTES,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "corpus_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "frozen_config.json").write_text(
        json.dumps(frozen_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "status": "complete",
                "output": str(output.relative_to(root)),
                "candidate_count": len(CANDIDATES),
                "row_level_values_persisted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
