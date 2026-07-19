"""Common-fold E2: frozen offline BanglaBERT over prompt/evidence-response pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .metrics import classification_metrics, predictions_from_label1
from .v5_lexical import build_v5_groups
from .v5_normalization import normalize_v5
from .v7_protocol import (
    FitScope,
    OUTER_FOLDS,
    OUTER_SEEDS,
    SelectionAudit,
    build_common_fold_assignments,
    calibration_table,
    has_context,
    make_inner_grouped_folds,
    metric_record,
    outer_indices,
    stable_training_row_ids,
    validate_probability_frame,
)


EXPERIMENT = "e2_frozen_evidence_conditioned_banglabert"
NO_EVIDENCE = "[NO_EVIDENCE]"
C_GRID = (0.1, 1.0, 10.0)
THRESHOLD_GRID = tuple(round(value / 100, 2) for value in range(40, 61, 2))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def semantic_pair(prompt: Any, evidence: Any, response: Any) -> tuple[str, str]:
    """Build the declared full-response pair; this performs no claim splitting."""
    prompt_text = normalize_v5(prompt)
    evidence_text = normalize_v5(evidence)
    if str(evidence).strip() == NO_EVIDENCE:
        evidence_text = NO_EVIDENCE
    sequence_a = f"[QUESTION] {prompt_text}\n[EVIDENCE] {evidence_text}"
    sequence_b = normalize_v5(response)
    return sequence_a, sequence_b


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def encode_pairs(
    sequence_a: Sequence[str],
    sequence_b: Sequence[str],
    *,
    checkpoint_dir: Path,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode locally with a frozen checkpoint and deterministic mean pooling."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint_dir, local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(sequence_a), batch_size):
            stop = min(start + batch_size, len(sequence_a))
            batch = tokenizer(
                list(sequence_a[start:stop]),
                list(sequence_b[start:stop]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            output = model(**batch).last_hidden_state
            pooled = _mean_pool(output, batch["attention_mask"])
            rows.append(pooled.cpu().numpy().astype(np.float32))
    result = np.concatenate(rows, axis=0)
    if result.shape[0] != len(sequence_a) or not np.isfinite(result).all():
        raise RuntimeError("Semantic embedding output is incomplete or non-finite")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_embedding_states(
    frame: pd.DataFrame,
    snippets: Sequence[str],
    presence: np.ndarray,
    *,
    checkpoint_dir: Path,
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute retrieved and rejected evidence states without using any labels."""
    accepted_a: list[str] = []
    accepted_b: list[str] = []
    rejected_a: list[str] = []
    rejected_b: list[str] = []
    absent_indices: list[int] = []
    accepted_lengths = np.zeros(len(frame), dtype=np.int64)
    rejected_lengths = np.zeros(len(frame), dtype=np.int64)
    for index, row in enumerate(frame.itertuples(index=False)):
        evidence = row.context if presence[index] else snippets[index]
        rejected_evidence = row.context if presence[index] else NO_EVIDENCE
        pair_a = semantic_pair(row.prompt_bn, evidence, row.response_bn)
        pair_b = semantic_pair(row.prompt_bn, rejected_evidence, row.response_bn)
        accepted_a.append(pair_a[0])
        accepted_b.append(pair_a[1])
        if not presence[index]:
            absent_indices.append(index)
            rejected_a.append(pair_b[0])
            rejected_b.append(pair_b[1])
        accepted_lengths[index] = len(normalize_v5(evidence))
        rejected_lengths[index] = len(normalize_v5(rejected_evidence))
    accepted = encode_pairs(
        accepted_a,
        accepted_b,
        checkpoint_dir=checkpoint_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    rejected = accepted.copy()
    rejected_absent = encode_pairs(
        rejected_a,
        rejected_b,
        checkpoint_dir=checkpoint_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    rejected[np.asarray(absent_indices, dtype=np.int64)] = rejected_absent
    return accepted, rejected, accepted_lengths, rejected_lengths


def _feature_state(
    accepted: np.ndarray,
    rejected: np.ndarray,
    similarities: np.ndarray,
    presence: np.ndarray,
    accepted_lengths: np.ndarray,
    rejected_lengths: np.ndarray,
    retrieval_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evidence_accepted = presence | (similarities >= float(retrieval_threshold))
    matrix = np.where(evidence_accepted[:, None], accepted, rejected)
    evidence_lengths = np.where(evidence_accepted, accepted_lengths, rejected_lengths)
    return matrix.astype(np.float64), evidence_accepted, evidence_lengths


def _fit_model(features: np.ndarray, labels: np.ndarray, classifier_c: float) -> dict[str, Any]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=float(classifier_c),
        class_weight="balanced",
        max_iter=3000,
        solver="liblinear",
        random_state=42,
    ).fit(scaler.transform(features), labels)
    return {"scaler": scaler, "classifier": classifier}


def _predict_model(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    classifier = model["classifier"]
    classes = list(classifier.classes_)
    probabilities = classifier.predict_proba(model["scaler"].transform(features))[
        :, classes.index(1)
    ]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Semantic classifier returned non-finite probabilities")
    return probabilities.astype(np.float64)


def select_inner_configuration(
    features: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    *,
    seed: int,
    outer_fold: int,
) -> tuple[float, float, dict[str, Any], np.ndarray]:
    inner_folds = make_inner_grouped_folds(
        labels, groups, outer_train, seed=seed, outer_fold=outer_fold
    )
    candidates: list[tuple[tuple[float, ...], float, float, dict[str, Any]]] = []
    probabilities_by_c: dict[float, np.ndarray] = {}
    for classifier_c in C_GRID:
        probabilities = np.full(len(labels), np.nan, dtype=np.float64)
        for split in inner_folds:
            inner_train = np.asarray(split.train_indices, dtype=np.int64)
            inner_validation = np.asarray(split.validation_indices, dtype=np.int64)
            FitScope(
                role="e2_inner_scaler_classifier_fit",
                training_row_ids=tuple(row_ids[inner_train]),
                validation_row_ids=tuple(row_ids[inner_validation]),
            ).validate()
            for selection_kind in ("model_hyperparameter", "classification_threshold"):
                SelectionAudit(
                    selection_kind=selection_kind,
                    inner_training_row_ids=tuple(row_ids[inner_train]),
                    inner_validation_row_ids=tuple(row_ids[inner_validation]),
                ).validate()
            model = _fit_model(features[inner_train], labels[inner_train], classifier_c)
            probabilities[inner_validation] = _predict_model(model, features[inner_validation])
        selected_probabilities = probabilities[outer_train]
        if not np.isfinite(selected_probabilities).all():
            raise RuntimeError("E2 inner cross-fit did not cover outer-training rows")
        probabilities_by_c[float(classifier_c)] = selected_probabilities.copy()
        for threshold in THRESHOLD_GRID:
            predictions = predictions_from_label1(selected_probabilities, threshold)
            metrics = classification_metrics(labels[outer_train], predictions)
            counts = np.bincount(predictions, minlength=2)
            maximum_share = float(counts.max() / counts.sum())
            if maximum_share > 0.90:
                continue
            record = {
                "classifier_c": classifier_c,
                "classification_threshold": threshold,
                **metrics,
                "maximum_predicted_class_share": maximum_share,
                "selection_scope": "outer_training_inner_grouped_oof_only",
            }
            rank = (
                float(metrics["macro_f1"]),
                float(metrics["f1_label0"]),
                -abs(float(threshold) - 0.5),
                -abs(float(np.log10(classifier_c))),
                -float(threshold),
            )
            candidates.append((rank, float(classifier_c), float(threshold), record))
    if not candidates:
        raise RuntimeError("Every predeclared E2 configuration collapsed")
    _, classifier_c, threshold, record = max(candidates, key=lambda item: item[0])
    return classifier_c, threshold, record, probabilities_by_c[classifier_c]


def _load_retrieval(cache_path: Path, expected_rows: int) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    payload = joblib.load(cache_path)
    snippets = list(payload["snippets"])
    similarities = np.asarray(payload["similarities"], dtype=np.float64)
    if len(snippets) != expected_rows or similarities.shape != (expected_rows,):
        raise RuntimeError("Retrieval cache is not aligned to the official training data")
    if int(payload.get("test_rows_accessed", -1)) != 0:
        raise RuntimeError("Retrieval cache provenance indicates test access")
    return snippets, similarities, dict(payload["resource"])


def _retrieval_thresholds(e0_dir: Path) -> dict[tuple[int, int], float]:
    table = pd.read_csv(e0_dir / "fold_metrics.csv")
    result = {
        (int(row.seed), int(row.outer_fold)): float(row.retrieval_threshold)
        for row in table.itertuples(index=False)
    }
    expected = {(seed, fold) for seed in OUTER_SEEDS for fold in range(1, OUTER_FOLDS + 1)}
    if set(result) != expected:
        raise RuntimeError("E0 does not provide one retrieval threshold per common outer fold")
    return result


def run_e2(
    train_path: Path,
    output_dir: Path,
    folds_path: Path,
    retrieval_cache: Path,
    e0_dir: Path,
    checkpoint_dir: Path,
    checkpoint_revision: str,
    checkpoint_weight_sha256: str,
    *,
    max_length: int,
    batch_size: int,
) -> dict[str, Any]:
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"E2 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir()

    weight_path = checkpoint_dir / "pytorch_model.bin"
    actual_weight_hash = sha256_file(weight_path)
    if actual_weight_hash != checkpoint_weight_sha256:
        raise RuntimeError("Pinned BanglaBERT weight SHA-256 does not match")
    frame = load_labeled_json(train_path, expected_sha256=OFFICIAL_SAMPLE_SHA256).reset_index(
        drop=True
    )
    labels = frame["label"].to_numpy(dtype=np.int64)
    presence = np.asarray([has_context(value) for value in frame["context"]], dtype=bool)
    row_ids = np.asarray(stable_training_row_ids(frame), dtype=object)
    group_audit = build_v5_groups(frame)
    groups = np.asarray(group_audit.group_ids, dtype=object)
    assignments = build_common_fold_assignments(frame)
    if folds_path.exists():
        existing = pd.read_csv(folds_path)
        if not assignments.equals(existing):
            raise RuntimeError("Frozen common-fold assignment differs from regenerated folds")
    else:
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(folds_path, index=False)

    snippets, similarities, resource = _load_retrieval(retrieval_cache, len(frame))
    retrieval_thresholds = _retrieval_thresholds(e0_dir)
    embedding_started = perf_counter()
    accepted, rejected, accepted_lengths, rejected_lengths = build_embedding_states(
        frame,
        snippets,
        presence,
        checkpoint_dir=checkpoint_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    embedding_seconds = perf_counter() - embedding_started
    np.savez_compressed(
        output_dir / "training_embedding_states.npz",
        accepted=accepted,
        rejected=rejected,
        row_ids=row_ids,
    )

    oof_records: list[pd.DataFrame] = []
    stack_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    model_manifest: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        for outer_fold in range(1, OUTER_FOLDS + 1):
            fold_started = perf_counter()
            outer_train, outer_validation = outer_indices(
                assignments, row_ids, seed=seed, outer_fold=outer_fold
            )
            FitScope(
                role="e2_outer_scaler_classifier_fit",
                training_row_ids=tuple(row_ids[outer_train]),
                validation_row_ids=tuple(row_ids[outer_validation]),
            ).validate()
            retrieval_threshold = retrieval_thresholds[(seed, outer_fold)]
            features, evidence_accepted, evidence_lengths = _feature_state(
                accepted,
                rejected,
                similarities,
                presence,
                accepted_lengths,
                rejected_lengths,
                retrieval_threshold,
            )
            classifier_c, threshold, inner_record, inner_probabilities = select_inner_configuration(
                features,
                labels,
                row_ids,
                groups,
                outer_train,
                seed=seed,
                outer_fold=outer_fold,
            )
            stack_records.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[outer_train],
                        "seed": seed,
                        "target_outer_fold": outer_fold,
                        "label": labels[outer_train],
                        "context_present": presence[outer_train],
                        "probability_label_1_e2": inner_probabilities,
                        "probability_label_0_e2": 1.0 - inner_probabilities,
                        "retrieval_score": similarities[outer_train],
                        "retrieval_accepted": evidence_accepted[outer_train],
                        "evidence_length": evidence_lengths[outer_train],
                        "retrieval_threshold_e0": retrieval_threshold,
                        "classification_threshold_e2": threshold,
                        "base_prediction_provenance": "outer_train_inner_grouped_cross_fit",
                    }
                )
            )
            model = _fit_model(features[outer_train], labels[outer_train], classifier_c)
            probabilities = _predict_model(model, features[outer_validation])
            model["metadata"] = {
                "classifier_c": classifier_c,
                "classification_threshold": threshold,
                "retrieval_threshold": retrieval_threshold,
                "checkpoint_revision": checkpoint_revision,
                "checkpoint_weight_sha256": actual_weight_hash,
                "probability_orientation": "probability_label_1_is_faithful",
            }
            model_path = models_dir / f"seed_{seed}_fold_{outer_fold}.joblib"
            joblib.dump(model, model_path, compress=3)
            reloaded = joblib.load(model_path)
            reload_probabilities = _predict_model(reloaded, features[outer_validation])
            reload_max_abs_diff = float(np.max(np.abs(reload_probabilities - probabilities)))
            if reload_max_abs_diff > 1e-12:
                raise RuntimeError("Reloaded E2 classifier predictions differ")
            predictions = predictions_from_label1(probabilities, threshold)
            fold_metrics.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "classifier_c": classifier_c,
                    "classification_threshold": threshold,
                    "retrieval_threshold": retrieval_threshold,
                    "inner_selection": inner_record,
                    "group_overlap_count": 0,
                    "runtime_seconds": perf_counter() - fold_started,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                    "reload_max_abs_probability_difference": reload_max_abs_diff,
                    **metric_record(
                        labels[outer_validation],
                        probabilities,
                        predictions,
                        presence[outer_validation],
                    ),
                }
            )
            oof_records.append(
                pd.DataFrame(
                    {
                        "row_id": row_ids[outer_validation],
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "label": labels[outer_validation],
                        "context_present": presence[outer_validation],
                        "probability_label_0": 1.0 - probabilities,
                        "probability_label_1": probabilities,
                        "classification_threshold": threshold,
                        "prediction": predictions,
                        "retrieval_score": similarities[outer_validation],
                        "retrieval_accepted": evidence_accepted[outer_validation],
                        "evidence_length": evidence_lengths[outer_validation],
                    }
                )
            )
            model_manifest.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "path": model_path.relative_to(output_dir).as_posix(),
                    "bytes": model_path.stat().st_size,
                    "sha256": sha256_file(model_path),
                    "reload_max_abs_probability_difference": reload_max_abs_diff,
                }
            )

    oof = pd.concat(oof_records, ignore_index=True).sort_values(
        ["seed", "outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    validate_probability_frame(oof, row_ids)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    stack_training = pd.concat(stack_records, ignore_index=True).sort_values(
        ["seed", "target_outer_fold", "row_id"], kind="mergesort"
    ).reset_index(drop=True)
    stack_training.to_csv(output_dir / "stack_training_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).drop(columns=["route_metrics", "inner_selection"]).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )

    seed_records: list[dict[str, Any]] = []
    for seed, selected in oof.groupby("seed", sort=True):
        seed_records.append(
            {
                "seed": int(seed),
                **metric_record(
                    selected["label"],
                    selected["probability_label_1"],
                    selected["prediction"],
                    selected["context_present"],
                ),
            }
        )
    pd.DataFrame(seed_records).drop(columns=["route_metrics"]).to_csv(
        output_dir / "seed_metrics.csv", index=False
    )
    aggregate = metric_record(
        oof["label"], oof["probability_label_1"], oof["prediction"], oof["context_present"]
    )
    pd.DataFrame(
        [
            {"route": name, **values}
            for name, values in aggregate["route_metrics"].items()
        ]
    ).to_csv(output_dir / "route_metrics.csv", index=False)
    summary = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "encoder_policy": "frozen_offline_mean_pooled",
        "probability_orientation": "probability_label_1_is_faithful_label_1",
        "official_rows": len(frame),
        "oof_rows": len(oof),
        "complete_oof_predictions_per_official_row": len(OUTER_SEEDS),
        "aggregate_repeated_oof": aggregate,
        "seed_metrics": seed_records,
        "fold_metric_distribution": {
            name: {
                "mean": float(np.mean([record[name] for record in fold_metrics])),
                "std": float(np.std([record[name] for record in fold_metrics], ddof=0)),
                "minimum": float(np.min([record[name] for record in fold_metrics])),
                "maximum": float(np.max([record[name] for record in fold_metrics])),
            }
            for name in ("macro_f1", "f1_label0", "f1_label1", "brier_score")
        },
        "calibration": calibration_table(oof["label"], oof["probability_label_1"]),
        "runtime_seconds": perf_counter() - started,
        "embedding_seconds": embedding_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "test_rows_accessed": 0,
        "reload_all_equal": True,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "confusion_matrices.json",
        {
            "aggregate": aggregate["confusion_matrix"],
            "routes": {
                name: values["confusion_matrix"]
                for name, values in aggregate["route_metrics"].items()
            },
        },
    )
    _write_json(
        output_dir / "config.json",
        {
            "experiment": EXPERIMENT,
            "checkpoint_revision": checkpoint_revision,
            "checkpoint_weight_sha256": actual_weight_hash,
            "local_files_only": True,
            "encoder_frozen": True,
            "pooling": "attention_masked_mean_last_hidden_state",
            "semantic_pair": {
                "sequence_a": "[QUESTION] prompt + [EVIDENCE] official_or_retrieved_evidence",
                "sequence_b": "full_normalized_response",
                "atomic_claim_splitting": False,
                "rejected_marker": NO_EVIDENCE,
            },
            "max_length": max_length,
            "batch_size": batch_size,
            "classifier_c_grid": C_GRID,
            "classification_threshold_grid": THRESHOLD_GRID,
            "retrieval_threshold_source": "E0 nested inner grouped selection for same outer fold",
            "selection_scope": "nested_inner_grouped_only",
            "test_data_role": "not_accessed",
        },
    )
    _write_json(
        output_dir / "checkpoint_manifest.json",
        {
            "model": "csebuetnlp/banglabert",
            "revision": checkpoint_revision,
            "weight_file": "pytorch_model.bin",
            "weight_bytes": weight_path.stat().st_size,
            "weight_sha256": actual_weight_hash,
            "authenticated": True,
            "local_files_only": True,
            "resource": resource,
            "retrieval_cache_sha256": sha256_file(retrieval_cache),
            "model_artifacts": model_manifest,
        },
    )
    _write_json(
        output_dir / "runtime.json",
        {
            "total_seconds": summary["runtime_seconds"],
            "embedding_seconds": embedding_seconds,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "peak_vram_bytes": summary["peak_vram_bytes"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scikit_learn": sklearn.__version__,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7/e2"))
    parser.add_argument("--folds-path", type=Path, default=Path("artifacts/v7/folds/common_folds.csv"))
    parser.add_argument("--retrieval-cache", type=Path, required=True)
    parser.add_argument("--e0-dir", type=Path, default=Path("artifacts/v7/e0"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-weight-sha256", required=True)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_e2(
        args.train_data,
        args.output_dir,
        args.folds_path,
        args.retrieval_cache,
        args.e0_dir,
        args.checkpoint_dir,
        args.checkpoint_revision,
        args.checkpoint_weight_sha256,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "experiment": summary["experiment"],
                "macro_f1": summary["aggregate_repeated_oof"]["macro_f1"],
                "test_rows_accessed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
