"""One-shot V14 correction for the official hallucinated-class F1 metric."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .champion_submission import EXPECTED_TEST_ROWS, validate_champion_test_frame
from .champion_submission_runner import authenticate_test_file, process_memory_bytes
from .data_loading import sha256_file
from .submission import build_submission
from .v4_preprocessing import official_context_is_present
from .v4_runner import load_official_training_frame
from .v5_lexical import deterministic_substring_prediction
from .v5_sparse import SparseNullModel
from .v7_corpus import load_corpus
from .v7_features import TOP5_FEATURES, V4A_FEATURES, feature_vector, top5_features, v4a_features
from .v7_retrieval import CharacterTfidfRetriever, RetrievedPassage
from .v8_route_complement import CANDIDATE_I, V4A_RETRIEVAL_CUTOFF, _build_v4a_classifier, build_v4a_feature_matrix
from .v9_features import retrieval_feature_matrix
from .v9_nli import run_frozen_nli_inference
from .v9_runner import _parallel_retrieve, memory_status
from .v10_ensemble import _pipeline as r_pipeline
from .v14_evaluation import run_v14_nested
from .v14_metrics import THRESHOLD_GRID, metric_record, predictions_from_label1, select_threshold


EXPERIMENT = "v14_hallucinated_class_f1_correction"
MODE = "metric-correction"
EXPECTED_CORPUS_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
MDEBERTA_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
MDEBERTA_SHA256 = "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen V14 metric correction cycle")
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["logistic"]
    return model.predict_proba(matrix)[:, list(classifier.classes_).index(1)].astype(np.float64)


def _unlabeled_retrieval_features(
    frame: pd.DataFrame, evidence: Sequence[Sequence[RetrievedPassage]]
) -> np.ndarray:
    rows = [
        feature_vector(top5_features(row.response_bn, evidence[position], cutoff=0.0), TOP5_FEATURES)
        for position, row in enumerate(frame.itertuples(index=False))
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(frame), len(TOP5_FEATURES)) or not np.isfinite(matrix).all():
        raise RuntimeError("V14 unlabeled retrieval features are invalid")
    return matrix


def _unlabeled_v4a_features(
    frame: pd.DataFrame,
    presence: np.ndarray,
    absent_evidence: Sequence[Sequence[RetrievedPassage]],
) -> np.ndarray:
    rows = []
    absent_position = 0
    for row, is_present in zip(frame.itertuples(index=False), presence, strict=True):
        if is_present:
            evidence = (RetrievedPassage("official-context-in-memory", "", str(row.context), 1.0),)
        else:
            evidence = absent_evidence[absent_position]
            absent_position += 1
        rows.append(feature_vector(v4a_features(row.response_bn, evidence, cutoff=V4A_RETRIEVAL_CUTOFF), V4A_FEATURES))
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(frame), len(V4A_FEATURES)) or not np.isfinite(matrix).all():
        raise RuntimeError("V14 unlabeled V4-A features are invalid")
    return matrix


def _artifact_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    matches = [name for name in candidates if name in frame.columns]
    if len(matches) != 1:
        raise ValueError(f"V14 expected exactly one artifact column from {tuple(candidates)}")
    return matches[0]


def unverified_v4a_crossfold_threshold_audit(
    official: pd.DataFrame, artifact_path: Path
) -> dict[str, Any]:
    artifact = pd.read_csv(artifact_path)
    probability_column = _artifact_column(artifact, ("probability_label1", "label_1_probability"))
    row_column = _artifact_column(artifact, ("row_index", "source_index"))
    fold_column = _artifact_column(artifact, ("fold", "fold_id"))
    if len(artifact) != len(official) or artifact[row_column].duplicated().any():
        raise ValueError("Unverified V4-A artifact row coverage is invalid")
    aligned = artifact.sort_values(row_column).reset_index(drop=True)
    if aligned[row_column].astype(int).tolist() != list(range(len(official))):
        raise ValueError("Unverified V4-A artifact row order cannot be authenticated")
    if "label" in aligned and aligned["label"].astype(int).tolist() != official["label"].astype(int).tolist():
        raise ValueError("Unverified V4-A artifact labels do not align")
    truth = official["label"].to_numpy(dtype=np.int64)
    probability = aligned[probability_column].to_numpy(dtype=np.float64)
    folds = aligned[fold_column].to_numpy(dtype=np.int64)
    if not np.isfinite(probability).all() or set(np.unique(folds)) != set(range(5)):
        raise ValueError("Unverified V4-A probabilities or fold identifiers are invalid")
    crossfold = np.full(len(truth), -1, dtype=np.int64)
    records = []
    for held_fold in range(5):
        training = folds != held_fold
        validation = ~training
        selected = select_threshold(truth[training], probability[training])["selected"]
        prediction = predictions_from_label1(probability[validation], selected["threshold"])
        fixed = predictions_from_label1(probability[validation], 0.50)
        selected_metrics = metric_record(truth[validation], prediction)
        fixed_metrics = metric_record(truth[validation], fixed)
        crossfold[validation] = prediction
        records.append(
            {
                "held_fold": held_fold,
                "selected_threshold": selected["threshold"],
                "selected_metrics": selected_metrics,
                "fixed_050_metrics": fixed_metrics,
                "label0_f1_improvement": selected_metrics["label0_f1"] - fixed_metrics["label0_f1"],
                "model_training_membership_verified": False,
            }
        )
    selected_metrics = metric_record(truth, crossfold)
    fixed_metrics = metric_record(truth, predictions_from_label1(probability, 0.50))
    thresholds = [record["selected_threshold"] for record in records]
    improvement = selected_metrics["label0_f1"] - fixed_metrics["label0_f1"]
    return {
        "name": "unverified_v4a_crossfold_threshold_audit",
        "artifact_sha256": sha256_file(artifact_path),
        "model_oof_exclusion_verified": False,
        "probability_orientation": "P(label 1)",
        "fixed_050": fixed_metrics,
        "crossfold": selected_metrics,
        "improvement": improvement,
        "folds_improved": int(sum(record["label0_f1_improvement"] > 0 for record in records)),
        "selected_thresholds": thresholds,
        "median_deployment_threshold": float(np.median(thresholds)),
        "boundary_concentrated": bool(sum(value in (THRESHOLD_GRID[0], THRESHOLD_GRID[-1]) for value in thresholds) >= 3),
        "local_gate_before_test_share": bool(improvement >= 0.015 and sum(record["label0_f1_improvement"] > 0 for record in records) >= 4 and sum(value in (THRESHOLD_GRID[0], THRESHOLD_GRID[-1]) for value in thresholds) < 3),
        "fold_records": records,
    }


def _deployment_thresholds(evaluation: dict[str, Any], candidate: str) -> dict[str, float]:
    records = [record["selected_thresholds"][candidate] for record in evaluation["outer_folds"]]
    keys = [key for key in records[0] if key.endswith("threshold")]
    return {key: float(np.median([float(record[key]) for record in records])) for key in keys}


def _full_data_predictions(
    candidate: str,
    thresholds: dict[str, float],
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_presence: np.ndarray,
    test_presence: np.ndarray,
    train_v4a: np.ndarray,
    test_v4a: np.ndarray,
    train_r: np.ndarray,
    test_r: np.ndarray | None,
) -> np.ndarray:
    labels = train["label"].to_numpy(dtype=np.int64)
    prediction = np.empty(len(test), dtype=np.int64)
    if candidate == "Y4":
        present = test.loc[test_presence]
        prediction[test_presence] = np.asarray(
            [deterministic_substring_prediction(row.context, row.response_bn) for row in present.itertuples(index=False)], dtype=np.int64
        )
    else:
        v4a = _build_v4a_classifier().fit(train_v4a, labels)
        v4a_probability = _probability(v4a, test_v4a)
        if candidate == "Y0":
            return predictions_from_label1(v4a_probability, thresholds["threshold"])
        prediction[test_presence] = predictions_from_label1(
            v4a_probability[test_presence], thresholds["present_threshold"]
        )
    absent_train = train.loc[~train_presence].reset_index(drop=True)
    absent_test = test.loc[~test_presence].reset_index(drop=True)
    if candidate in ("Y1", "Y4"):
        model = SparseNullModel(CANDIDATE_I).fit(absent_train)
        absent_probability = model.predict_unlabeled_label1_probability(absent_test)
    else:
        if test_r is None:
            raise RuntimeError("V14 Candidate R/U test features were not prepared")
        i_model = SparseNullModel(CANDIDATE_I).fit(absent_train)
        probability_i = i_model.predict_unlabeled_label1_probability(absent_test)
        r_model = r_pipeline().fit(train_r[~train_presence], labels[~train_presence])
        probability_r = _probability(r_model, test_r[~test_presence])
        absent_probability = probability_r if candidate == "Y2" else 0.5 * (probability_i + probability_r)
    prediction[~test_presence] = predictions_from_label1(absent_probability, thresholds["absent_threshold"])
    return prediction


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_parser().parse_args(argv)
    root = repository_root()
    expected_output = (root / "artifacts" / "submissions" / "v14_metric_corrected").resolve()
    if not args.output_dir.is_absolute() or args.output_dir.resolve() != expected_output:
        raise ValueError("V14 output must be the approved absolute v14_metric_corrected directory")
    if expected_output.exists():
        raise ValueError("V14 output already exists; overwrite is prohibited")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    train = load_official_training_frame(root).reset_index(drop=True)
    train_presence = train["context"].map(official_context_is_present).to_numpy(dtype=bool)
    articles, corpus_manifest = load_corpus(root / "data" / "retrieval" / "bnwiki")
    if corpus_manifest.logical_content_manifest_sha256 != EXPECTED_CORPUS_MANIFEST:
        raise RuntimeError("V14 corpus authentication failed")
    retriever = CharacterTfidfRetriever(articles)
    absent_train = train.loc[~train_presence].reset_index(drop=True)
    train_evidence = _parallel_retrieve(retriever, absent_train["prompt_bn"].astype(str).tolist(), workers=4)
    train_v4a = build_v4a_feature_matrix(train, train_evidence)
    retrieval = retrieval_feature_matrix(absent_train, train_evidence)
    nli = run_frozen_nli_inference(
        root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{MDEBERTA_REVISION}",
        train_evidence,
        absent_train["response_bn"].astype(str).tolist(),
        use_fast=True,
    )
    if nli.identity.revision != MDEBERTA_REVISION or nli.identity.weight_sha256 != MDEBERTA_SHA256:
        raise RuntimeError("V14 mDeBERTa authentication failed")
    train_r = np.zeros((len(train), nli.features.shape[1] + retrieval.shape[1]), dtype=np.float64)
    train_r[~train_presence] = np.hstack((nli.features, retrieval))
    evaluation = run_v14_nested(train, train_v4a, train_r)
    teammate_path = root / "artifacts" / "shared" / "v4a_0685" / "v4a_oof_probabilities.csv"
    calibration = unverified_v4a_crossfold_threshold_audit(train, teammate_path)
    scientific = evaluation["selection"]
    output_files: list[Path] = []
    test_record: dict[str, Any] = {"opened": False}
    test: pd.DataFrame | None = None
    test_presence: np.ndarray | None = None
    test_evidence: Sequence[Sequence[RetrievedPassage]] | None = None
    test_v4a: np.ndarray | None = None
    test_r: np.ndarray | None = None
    if scientific["submission_worthy"] or calibration["local_gate_before_test_share"]:
        test_path = root / "data" / "competition" / "test set.csv"
        test_authentication = authenticate_test_file(test_path)
        test = validate_champion_test_frame(pd.read_csv(test_path))
        test_presence = test["context"].map(official_context_is_present).to_numpy(dtype=bool)
        if (int(test_presence.sum()), int((~test_presence).sum())) != (1361, 1155):
            raise RuntimeError("V14 authenticated competition-test route totals changed")
        test_record = {"opened": True, "authentication": test_authentication, "rows": len(test), "route_counts": {"present": int(test_presence.sum()), "absent": int((~test_presence).sum())}}
    expected_output.mkdir(parents=True, exist_ok=False)
    if scientific["submission_worthy"]:
        assert test is not None and test_presence is not None
        selected = str(scientific["selected_candidate"])
        thresholds = _deployment_thresholds(evaluation, selected)
        absent_test = test.loc[~test_presence].reset_index(drop=True)
        if selected != "Y4":
            test_evidence = _parallel_retrieve(retriever, absent_test["prompt_bn"].astype(str).tolist(), workers=4)
            test_v4a = _unlabeled_v4a_features(test, test_presence, test_evidence)
        else:
            test_v4a = np.empty((len(test), len(V4A_FEATURES)), dtype=np.float64)
        if selected in ("Y2", "Y3"):
            assert test_evidence is not None
            test_retrieval = _unlabeled_retrieval_features(absent_test, test_evidence)
            test_nli = run_frozen_nli_inference(
                root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{MDEBERTA_REVISION}",
                test_evidence,
                absent_test["response_bn"].astype(str).tolist(), use_fast=True,
            )
            test_r = np.zeros((len(test), test_nli.features.shape[1] + test_retrieval.shape[1]), dtype=np.float64)
            test_r[~test_presence] = np.hstack((test_nli.features, test_retrieval))
        predictions = _full_data_predictions(selected, thresholds, train, test, train_presence, test_presence, train_v4a, test_v4a, train_r, test_r)
        path = expected_output / "submission_v14_metric_corrected.csv"
        build_submission(test["id"], predictions).to_csv(path, index=False)
        output_files.append(path)
        scientific["deployment"] = {"thresholds": thresholds, "predicted_counts": metric_record(predictions, predictions)["predicted_counts"], "path": str(path)}
    if calibration["local_gate_before_test_share"]:
        assert test is not None
        test_artifact_path = root / "artifacts" / "shared" / "v4a_0685" / "v4a_test_probabilities.csv"
        test_artifact = pd.read_csv(test_artifact_path)
        probability_column = _artifact_column(test_artifact, ("probability_label1", "label_1_probability"))
        if len(test_artifact) != EXPECTED_TEST_ROWS:
            raise ValueError("Unverified V4-A test probability row count changed")
        if "id" in test_artifact and test_artifact["id"].tolist() != test["id"].tolist():
            raise ValueError("Unverified V4-A test probability ID order changed")
        predictions = predictions_from_label1(
            test_artifact[probability_column].to_numpy(dtype=np.float64), calibration["median_deployment_threshold"]
        )
        counts = np.bincount(predictions, minlength=2)
        calibration["test_predicted_counts"] = {"0": int(counts[0]), "1": int(counts[1])}
        calibration["test_label0_share"] = float(counts[0] / len(predictions))
        calibration["test_share_gate"] = bool(counts[0] / len(predictions) < 0.90)
        if calibration["test_share_gate"]:
            path = expected_output / "submission_v14_v4a_rethresholded.csv"
            build_submission(test["id"], predictions).to_csv(path, index=False)
            output_files.append(path)
            calibration["submission_path"] = str(path)
    resources = {
        "runtime_seconds": perf_counter() - started,
        "process_memory": process_memory_bytes(),
        "system_memory": memory_status(),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "retrieval_workers": 4,
        "classifier_workers": 1,
        "gpu_models_concurrent": 1,
    }
    result = {
        "experiment": EXPERIMENT,
        "metric_authentication": {"primary_metric": "binary F1 for hallucinated class", "hallucination_label": 0, "scorer": "f1_score(pos_label=0, zero_division=0)"},
        "evaluation": evaluation,
        "unverified_v4a_crossfold_threshold_audit": calibration,
        "test": test_record,
        "model_authentication": {**nli.identity.__dict__, "diagnostics": nli.diagnostics},
        "corpus_manifest_sha256": corpus_manifest.logical_content_manifest_sha256,
        "resources": resources,
        "safety": {"public_labeled_data_used": False, "leaderboard_used_for_selection": False, "raw_test_text_persisted": False, "test_labels_accessed": False},
    }
    summary_path = expected_output / "run_summary.json"
    _write_json(summary_path, result)
    output_files.append(summary_path)
    checksums_path = expected_output / "artifact_checksums.json"
    checksums = {path.name: sha256_file(path) for path in output_files}
    _write_json(checksums_path, checksums)
    print(json.dumps({"status": "complete", "selected": scientific["selected_candidate"], "submission_worthy": scientific["submission_worthy"], "outputs": [str(path) for path in output_files], "runtime_seconds": resources["runtime_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
