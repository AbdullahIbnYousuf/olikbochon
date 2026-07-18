"""Authenticated shared-fold OOF evaluation for the frozen V8 route complement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loading import validate_labeled_frame
from .metrics import classification_metrics, predictions_from_label1, subgroup_metrics
from .v4_preprocessing import official_context_is_present
from .v5_lexical import build_v5_folds, fold_feasibility_audit
from .v5_sparse import LOGISTIC_CONFIG, SparseNullModel
from .v7_features import V4A_FEATURES, feature_vector, v4a_features
from .v7_retrieval import RetrievedPassage


EXPERIMENT_NAME = "v8_authenticated_route_complement"
V4A_RETRIEVAL_CUTOFF = 0.25
CLASSIFIER_THRESHOLD = 0.50
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169
CANDIDATE_I = "candidate_i_sparse_lexical_union"
TEAMMATE_REFERENCES = {
    "v4a_full_macro_f1": 0.6872701508,
    "v4a_context_present_macro_f1": 0.9246667954,
    "v4a_context_absent_macro_f1": 0.4339712919,
    "candidate_i_context_absent_macro_f1": 0.534500,
    "routed_champion_macro_f1": 0.692051,
}
VERDICT_GATES = {
    "reject_at_or_below": 0.692051,
    "valid_improvement_above": 0.692051,
    "meaningful_at_least": 0.70,
    "strong_at_least": 0.72,
    "high_value_at_least": 0.74,
    "maximum_seed_std": 0.04,
    "maximum_class_share": 0.90,
}


def frozen_v8_config() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "routes": {
            "context_present": "regenerated_v4a_style_oof_probability_ge_0.50",
            "context_absent": "regenerated_candidate_i_oof_probability_ge_0.50",
        },
        "threshold": CLASSIFIER_THRESHOLD,
        "v4a": {
            "retrieval_cutoff": V4A_RETRIEVAL_CUTOFF,
            "top_k": 1,
            "snippet": "first_800_normalized_characters",
            "features": list(V4A_FEATURES),
            "standard_scaler": True,
            "scaler_evidence": (
                "repository V7 Candidate M was predeclared as standardized V4-A parity; "
                "teammate summary omitted scaler metadata"
            ),
            "classifier": LOGISTIC_CONFIG,
        },
        "candidate_i": {
            "candidate": CANDIDATE_I,
            "classifier": LOGISTIC_CONFIG,
            "fit_route": "current_outer_training_context_absent_rows_only",
        },
        "selection": "none_fixed_route_complement",
        "verdict_gates": VERDICT_GATES,
        "teammate_artifact_use": "post_hoc_aggregate_reference_only_not_scoring",
    }


def configuration_fingerprint() -> str:
    payload = json.dumps(
        frozen_v8_config(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_v4a_feature_matrix(
    frame: pd.DataFrame,
    absent_evidence: Sequence[Sequence[RetrievedPassage]],
) -> np.ndarray:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V8 corrected route totals differ from 130/169")
    if len(absent_evidence) != EXPECTED_ABSENT_ROWS:
        raise ValueError("V8 retrieval evidence must align to exactly 169 absent rows")
    absent_position = 0
    rows: list[np.ndarray] = []
    for row, is_present in zip(validated.itertuples(index=False), presence, strict=True):
        if is_present:
            passages = (
                RetrievedPassage(
                    "official-context-in-memory",
                    "",
                    str(row.context),
                    1.0,
                ),
            )
        else:
            passages = tuple(absent_evidence[absent_position])
            absent_position += 1
        record = v4a_features(
            row.response_bn,
            passages,
            cutoff=V4A_RETRIEVAL_CUTOFF,
        )
        rows.append(feature_vector(record, V4A_FEATURES))
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(validated), len(V4A_FEATURES)) or not np.isfinite(matrix).all():
        raise RuntimeError("V8 V4-A feature matrix is invalid")
    return matrix


def _index_digest(indices: np.ndarray) -> str:
    payload = ",".join(str(int(value)) for value in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _group_hashes(groups: Sequence[str]) -> list[str]:
    return [
        hashlib.sha256(str(group).encode("utf-8")).hexdigest()
        for group in sorted(set(groups))
    ]


def _update_array(digest: Any, name: str, values: Any) -> None:
    array = np.asarray(values)
    digest.update(name.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.astype(np.float64, copy=False).tobytes(order="C"))


def _v4a_model_fingerprint(model: Pipeline) -> str:
    digest = hashlib.sha256()
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["logistic"]
    for name in ("mean_", "scale_", "var_"):
        _update_array(digest, f"scaler.{name}", getattr(scaler, name))
    for name in ("classes_", "coef_", "intercept_", "n_iter_"):
        _update_array(digest, f"classifier.{name}", getattr(classifier, name))
    return digest.hexdigest()


def _candidate_i_model_fingerprint(model: SparseNullModel) -> str:
    digest = hashlib.sha256()
    for name, vectorizer in sorted(model.vectorizers.items()):
        digest.update(name.encode("ascii"))
        digest.update(
            json.dumps(
                sorted(
                    (str(token), int(index))
                    for token, index in vectorizer.vocabulary_.items()
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        _update_array(digest, f"{name}.idf", vectorizer.idf_)
    if model.numeric_extractor is None or model.numeric_scaler is None:
        raise RuntimeError("Candidate I fitted state is incomplete")
    digest.update(
        json.dumps(
            sorted(
                (str(token), int(count))
                for token, count in model.numeric_extractor.response_token_frequency.items()
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in ("mean_", "scale_", "var_"):
        _update_array(digest, f"numeric_scaler.{name}", getattr(model.numeric_scaler, name))
    for name in ("classes_", "coef_", "intercept_", "n_iter_"):
        _update_array(digest, f"classifier.{name}", getattr(model.classifier, name))
    return digest.hexdigest()


def _build_v4a_classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(**LOGISTIC_CONFIG)),
        ]
    )


def _prediction_counts(predictions: np.ndarray) -> dict[str, int]:
    counts = np.bincount(predictions, minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def _metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    presence: np.ndarray | None = None,
) -> dict[str, Any]:
    result = {
        **classification_metrics(truth, predictions),
        "predicted_counts": _prediction_counts(predictions),
    }
    if presence is not None:
        result["route_metrics"] = subgroup_metrics(truth, predictions, presence)
    return result


def _seed_distribution(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([record["macro_f1"] for record in records], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def run_authenticated_oof(
    frame: pd.DataFrame,
    v4a_matrix: np.ndarray,
    *,
    corpus_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validated = validate_labeled_frame(frame).reset_index(drop=True)
    if v4a_matrix.shape != (len(validated), len(V4A_FEATURES)):
        raise ValueError("V8 features do not align to the official frame")
    labels = validated["label"].to_numpy(dtype=np.int64)
    presence = np.asarray(
        [official_context_is_present(value) for value in validated["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V8 corrected route totals differ from 130/169")
    folds = build_v5_folds(validated)
    feasibility = fold_feasibility_audit(validated, folds)
    groups = np.asarray(folds.audit.group_ids, dtype=object)
    config_fingerprint = configuration_fingerprint()
    per_seed_v4a = {
        seed: np.full(len(validated), np.nan, dtype=np.float64) for seed in folds.seeds
    }
    per_seed_candidate_i = {
        seed: np.full(len(validated), np.nan, dtype=np.float64) for seed in folds.seeds
    }
    provenance_records: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray(split.validation_indices, dtype=np.int64)
        train_groups = groups[train]
        validation_groups = groups[validation]
        overlap = set(train_groups) & set(validation_groups)
        if overlap or set(train) & set(validation):
            raise RuntimeError("V8 fold has training/validation overlap")
        v4a_model = _build_v4a_classifier().fit(v4a_matrix[train], labels[train])
        v4a_classes = list(v4a_model.named_steps["logistic"].classes_)
        v4a_probabilities = v4a_model.predict_proba(v4a_matrix[validation])[
            :, v4a_classes.index(1)
        ].astype(np.float64)
        train_absent = train[~presence[train]]
        validation_absent = validation[~presence[validation]]
        candidate_i = SparseNullModel(CANDIDATE_I).fit(
            validated.iloc[train_absent].reset_index(drop=True)
        )
        candidate_i_probabilities = candidate_i.predict_label1_probability(
            validated.iloc[validation_absent].reset_index(drop=True)
        )
        if not np.isfinite(v4a_probabilities).all() or not np.isfinite(
            candidate_i_probabilities
        ).all():
            raise RuntimeError("V8 produced NaN or Inf probabilities")
        per_seed_v4a[split.seed][validation] = v4a_probabilities
        per_seed_candidate_i[split.seed][validation_absent] = candidate_i_probabilities
        validation_v4a = predictions_from_label1(
            v4a_probabilities, CLASSIFIER_THRESHOLD
        )
        validation_q = validation_v4a.copy()
        validation_q[~presence[validation]] = predictions_from_label1(
            candidate_i_probabilities, CLASSIFIER_THRESHOLD
        )
        provenance_records.append(
            {
                "seed": split.seed,
                "fold": split.fold,
                "validation_row_indices": [int(value) for value in validation],
                "training_row_count": int(train.size),
                "validation_row_count": int(validation.size),
                "candidate_i_training_row_count": int(train_absent.size),
                "candidate_i_validation_row_count": int(validation_absent.size),
                "training_group_hashes": _group_hashes(train_groups),
                "validation_group_hashes": _group_hashes(validation_groups),
                "group_overlap_count": 0,
                "row_overlap_count": 0,
                "training_index_sha256": _index_digest(train),
                "validation_index_sha256": _index_digest(validation),
                "model_configuration_fingerprint": config_fingerprint,
                "corpus_manifest_fingerprint": corpus_manifest_sha256,
                "v4a_fitted_model_fingerprint": _v4a_model_fingerprint(v4a_model),
                "candidate_i_fitted_model_fingerprint": _candidate_i_model_fingerprint(
                    candidate_i
                ),
                "v4a_preprocessing_fit_rows": int(train.size),
                "candidate_i_preprocessing_fit_rows": int(train_absent.size),
                "validation_rows_used_for_fitted_preprocessing": 0,
                "probability_label": 1,
                "threshold": CLASSIFIER_THRESHOLD,
            }
        )
        fold_metrics.append(
            {
                "seed": split.seed,
                "fold": split.fold,
                "v4a": _metrics(labels[validation], validation_v4a, presence[validation]),
                "candidate_i_absent": _metrics(
                    labels[validation_absent],
                    predictions_from_label1(
                        candidate_i_probabilities, CLASSIFIER_THRESHOLD
                    ),
                ),
                "candidate_q": _metrics(
                    labels[validation], validation_q, presence[validation]
                ),
            }
        )
    seed_v4a_records: list[dict[str, Any]] = []
    seed_i_records: list[dict[str, Any]] = []
    seed_q_records: list[dict[str, Any]] = []
    all_v4a_probabilities: list[np.ndarray] = []
    all_i_probabilities: list[np.ndarray] = []
    all_v4a_predictions: list[np.ndarray] = []
    all_i_predictions: list[np.ndarray] = []
    all_q_predictions: list[np.ndarray] = []
    for seed in folds.seeds:
        v4a_probabilities = per_seed_v4a[seed]
        candidate_i_probabilities = per_seed_candidate_i[seed][~presence]
        if not np.isfinite(v4a_probabilities).all() or not np.isfinite(
            candidate_i_probabilities
        ).all():
            raise RuntimeError("V8 OOF coverage is incomplete")
        v4a_predictions = predictions_from_label1(
            v4a_probabilities, CLASSIFIER_THRESHOLD
        )
        candidate_i_predictions = predictions_from_label1(
            candidate_i_probabilities, CLASSIFIER_THRESHOLD
        )
        q_predictions = v4a_predictions.copy()
        q_predictions[~presence] = candidate_i_predictions
        seed_v4a_records.append(_metrics(labels, v4a_predictions, presence))
        seed_i_records.append(_metrics(labels[~presence], candidate_i_predictions))
        seed_q_records.append(_metrics(labels, q_predictions, presence))
        all_v4a_probabilities.append(v4a_probabilities)
        all_i_probabilities.append(candidate_i_probabilities)
        all_v4a_predictions.append(v4a_predictions)
        all_i_predictions.append(candidate_i_predictions)
        all_q_predictions.append(q_predictions)
    repeated_truth = np.tile(labels, len(folds.seeds))
    repeated_presence = np.tile(presence, len(folds.seeds))
    repeated_absent_truth = np.tile(labels[~presence], len(folds.seeds))
    v4a_probability_values = np.concatenate(all_v4a_probabilities)
    candidate_i_probability_values = np.concatenate(all_i_probabilities)
    v4a_metrics = {
        **_metrics(
            repeated_truth,
            np.concatenate(all_v4a_predictions),
            repeated_presence,
        ),
        "brier_score": float(
            np.mean(np.square(v4a_probability_values - repeated_truth))
        ),
        "seed_macro_f1_distribution": _seed_distribution(seed_v4a_records),
    }
    candidate_i_metrics = {
        **_metrics(repeated_absent_truth, np.concatenate(all_i_predictions)),
        "brier_score": float(
            np.mean(
                np.square(candidate_i_probability_values - repeated_absent_truth)
            )
        ),
        "seed_macro_f1_distribution": _seed_distribution(seed_i_records),
    }
    q_predictions = np.concatenate(all_q_predictions)
    q_metrics = {
        **_metrics(repeated_truth, q_predictions, repeated_presence),
        "seed_macro_f1_distribution": _seed_distribution(seed_q_records),
        "maximum_predicted_class_share": float(
            np.bincount(q_predictions, minlength=2).max() / len(q_predictions)
        ),
    }
    q_macro = float(q_metrics["macro_f1"])
    seed_std = float(q_metrics["seed_macro_f1_distribution"]["std"])
    maximum_share = float(q_metrics["maximum_predicted_class_share"])
    valid = (
        q_macro > VERDICT_GATES["valid_improvement_above"]
        and seed_std <= VERDICT_GATES["maximum_seed_std"]
        and maximum_share <= VERDICT_GATES["maximum_class_share"]
    )
    if not valid:
        verdict = "rejected"
    elif q_macro >= VERDICT_GATES["high_value_at_least"]:
        verdict = "high_value"
    elif q_macro >= VERDICT_GATES["strong_at_least"]:
        verdict = "strong"
    elif q_macro >= VERDICT_GATES["meaningful_at_least"]:
        verdict = "meaningful"
    else:
        verdict = "valid_improvement"
    results = {
        "experiment": EXPERIMENT_NAME,
        "route_counts": {
            "context_present": int(presence.sum()),
            "context_absent": int((~presence).sum()),
        },
        "validation": feasibility,
        "fold_metrics": fold_metrics,
        "regenerated_v4a": v4a_metrics,
        "regenerated_candidate_i": candidate_i_metrics,
        "candidate_q_authenticated": q_metrics,
        "teammate_reference_comparison": {
            **TEAMMATE_REFERENCES,
            "v4a_full_difference": float(
                v4a_metrics["macro_f1"] - TEAMMATE_REFERENCES["v4a_full_macro_f1"]
            ),
            "v4a_present_difference": float(
                v4a_metrics["route_metrics"]["context_present"]["macro_f1"]
                - TEAMMATE_REFERENCES["v4a_context_present_macro_f1"]
            ),
            "v4a_absent_difference": float(
                v4a_metrics["route_metrics"]["context_absent"]["macro_f1"]
                - TEAMMATE_REFERENCES["v4a_context_absent_macro_f1"]
            ),
            "candidate_i_absent_difference": float(
                candidate_i_metrics["macro_f1"]
                - TEAMMATE_REFERENCES["candidate_i_context_absent_macro_f1"]
            ),
        },
        "verdict": verdict,
        "verdict_gates": VERDICT_GATES,
        "genuine_oof_coverage": True,
        "group_overlap_count": 0,
        "row_overlap_count": 0,
        "probability_label": 1,
        "threshold": CLASSIFIER_THRESHOLD,
        "row_level_probabilities_persisted": False,
        "raw_text_persisted": False,
        "competition_test_accessed": False,
    }
    provenance = {
        "experiment": EXPERIMENT_NAME,
        "model_configuration_fingerprint": config_fingerprint,
        "corpus_manifest_fingerprint": corpus_manifest_sha256,
        "split_strategy": folds.split_strategy,
        "seeds": list(folds.seeds),
        "fold_count_per_seed": 5,
        "official_group_count": folds.audit.group_count,
        "folds": provenance_records,
        "each_validation_row_covered_once_per_seed": True,
        "raw_text_persisted": False,
    }
    return results, provenance
