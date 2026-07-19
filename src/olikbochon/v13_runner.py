"""CLI for the one frozen V13 public template-transfer development cycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from .champion_submission import validate_champion_test_frame
from .champion_submission_runner import authenticate_test_file, process_memory_bytes
from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .metrics import classification_metrics
from .submission import build_submission
from .v13_public_models import (
    PublicLinearClassifier,
    infer_banglabert_test,
    probe_batch_size,
    train_public_banglabert,
)
from .v13_public_pool import (
    add_keys,
    authenticate_public,
    conflict_audit,
    deduplicate_pool,
    official_safe_pool,
    overlap_audit,
    pool_manifest,
)
from .v13_template_transfer import FrozenTemplateTransfer
from .v4_preprocessing import official_context_is_present
from .v9_runner import memory_status


MODE = "public-template-transfer"
OUTPUT_NAME = "public_template_transfer"
MODEL_REVISION = "9ce791f330578f50da6bc52b54205166fb5d1c8c"
V4A_OOF_SHA256 = "78ee80a3f9944500090342170ccf60db7cbc793c7fbe6a17ad3e6a6906d51ea6"
V4A_TEST_SHA256 = "2dd5ffa97c285727143fefe7d172bd02588368b7c3ff111088709ce0eb27adc2"
TRANSFER_ROUTES = ("exact_pair", "skeleton", "high_confidence_nn")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final frozen V13 transfer cycle.")
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--train-public-path", required=True, type=Path)
    parser.add_argument("--validation-public-path", required=True, type=Path)
    parser.add_argument("--contrastive-public-path", required=True, type=Path)
    parser.add_argument("--test-path", required=True, type=Path)
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    return parser


def _metric(truth: np.ndarray, prediction: np.ndarray, presence: np.ndarray | None = None) -> dict[str, Any]:
    counts = np.bincount(prediction.astype(np.int64), minlength=2)
    result = {
        **classification_metrics(truth, prediction),
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "maximum_class_share": float(counts.max() / len(prediction)),
    }
    if presence is not None:
        result["context_present_macro_f1"] = classification_metrics(truth[presence], prediction[presence])["macro_f1"]
        result["context_absent_macro_f1"] = classification_metrics(truth[~presence], prediction[~presence])["macro_f1"]
    return result


def _transfer_metrics(truth: np.ndarray, predictions: np.ndarray, routes: np.ndarray) -> dict[str, Any]:
    transferred = np.isin(routes, TRANSFER_ROUTES)
    return {
        "coverage_count": int(transferred.sum()),
        "coverage_fraction": float(transferred.mean()),
        "precision": float(np.mean(predictions[transferred] == truth[transferred])) if transferred.any() else None,
        "inside": _metric(truth[transferred], predictions[transferred]) if transferred.any() else None,
        "outside": _metric(truth[~transferred], predictions[~transferred]) if (~transferred).any() else None,
        "route_counts": {str(key): int(value) for key, value in pd.Series(routes).value_counts().items()},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_parser().parse_args(argv)
    if args.workers != 6:
        raise ValueError("V13 requires six normalization and similarity workers")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"{variable} must equal 1")
    root = repository_root()
    expected = (
        root / "data" / "public_5k" / "bangla_hallucination_5k_train.json",
        root / "data" / "public_5k" / "bangla_hallucination_5k_validation.json",
        root / "data" / "public_5k" / "bangla_hallucination_5k_contrastive.json",
        root / "data" / "competition" / "test set.csv",
        root / "artifacts" / "v13" / OUTPUT_NAME,
        root / "artifacts" / "submissions" / "v13_public_template_transfer",
    )
    supplied = (
        args.train_public_path, args.validation_public_path, args.contrastive_public_path,
        args.test_path, args.output_dir, args.submission_dir,
    )
    if any(not path.is_absolute() for path in supplied) or any(path.resolve() != target.resolve() for path, target in zip(supplied, expected, strict=True)):
        raise ValueError("V13 paths differ from the approved authenticated paths")
    resumable_checkpoints = [
        args.output_dir / "checkpoints" / f"seed{seed}.pt" for seed in (17, 29, 43)
    ]
    resume_after_finalization_failure = (
        args.output_dir.is_dir()
        and all(path.is_file() for path in resumable_checkpoints)
        and {path.resolve() for path in args.output_dir.rglob("*") if path.is_file()}
        == {path.resolve() for path in resumable_checkpoints}
    )
    if (args.output_dir.exists() and not resume_after_finalization_failure) or args.submission_dir.exists():
        raise ValueError("V13 output directories already exist and are not resumable")

    train, train_audit = authenticate_public(args.train_public_path, workers=6)
    validation, validation_audit = authenticate_public(args.validation_public_path, workers=6)
    contrastive, contrastive_audit = authenticate_public(args.contrastive_public_path, workers=6)
    public_pool, dedup = deduplicate_pool(train, contrastive, validation)
    if contrastive_audit["label_counts"] != {"0": 2500, "1": 2500}:
        raise RuntimeError("V13 contrastive label semantics are incompatible")
    official = add_keys(
        load_labeled_json(root / "data" / "competition" / "dataset samples.json", expected_sha256=OFFICIAL_SAMPLE_SHA256),
        workers=6,
    )
    safe_pool, official_exclusions = official_safe_pool(public_pool, official)
    if not safe_pool.pair_key.isin(set(official.pair_key)).sum() == 0 or not safe_pool.skeleton_key.isin(set(official.skeleton_key)).sum() == 0:
        raise RuntimeError("V13 official contamination exclusion failed")
    public_audits = {
        "files": [train_audit, validation_audit, contrastive_audit],
        "contrastive_included": True,
        "deduplication": dedup,
        "official_safe_pool": official_exclusions,
        "conflicts": {
            "train_pair": conflict_audit(train, "pair_key"),
            "validation_pair": conflict_audit(validation, "pair_key"),
            "contrastive_pair": conflict_audit(contrastive, "pair_key"),
            "combined_skeleton": conflict_audit(pd.concat((train, contrastive), ignore_index=True), "skeleton_key"),
        },
        "overlap": {
            "train_validation": overlap_audit(train, validation),
            "contrastive_validation": overlap_audit(contrastive, validation),
            "public_official": overlap_audit(public_pool, official),
        },
        "provenance_verdict": "compatible public/synthetic labeled resource; no ID column or explicit competition-test label mapping; unknown upstream license retained as caveat",
    }

    transfer = FrozenTemplateTransfer(safe_pool)
    calibration = transfer.calibrate(validation)
    validation_xa_output = transfer.predict(validation)
    validation_truth = validation.label.to_numpy(dtype=np.int64)
    validation_xa = {
        **_metric(validation_truth, validation_xa_output.predictions),
        "transfer": _transfer_metrics(validation_truth, validation_xa_output.predictions, validation_xa_output.routes),
    }
    xb = PublicLinearClassifier().fit(safe_pool)
    validation_xb_prediction = xb.predict(validation)
    validation_xb = _metric(validation_truth, validation_xb_prediction)
    official_truth = official.label.to_numpy(dtype=np.int64)
    official_presence = np.asarray(
        [official_context_is_present(value) for value in official.context], dtype=bool
    )
    official_xa_output = transfer.predict(official)
    official_xa = {
        **_metric(official_truth, official_xa_output.predictions, official_presence),
        "transfer": _transfer_metrics(official_truth, official_xa_output.predictions, official_xa_output.routes),
    }
    official_xb_prediction = xb.predict(official)
    official_xb = _metric(official_truth, official_xb_prediction, official_presence)

    model_path = root / "data" / "models" / f"banglabert-official-{MODEL_REVISION[:7]}"
    if resume_after_finalization_failure:
        batch_size, probe = probe_batch_size(model_path)
        validation_probability = infer_banglabert_test(
            validation, model_path=model_path,
            checkpoint_paths=[str(path) for path in resumable_checkpoints],
            batch_size=batch_size,
        )
        official_probability = infer_banglabert_test(
            official, model_path=model_path,
            checkpoint_paths=[str(path) for path in resumable_checkpoints],
            batch_size=batch_size,
        )
        neural = type("RecoveredNeuralResult", (), {
            "validation_probability": validation_probability,
            "official_probability": official_probability,
            "seed_records": [
                {"seed": seed, "selected_checkpoint_reused": True, "selected_epoch": None}
                for seed in (17, 29, 43)
            ],
            "checkpoint_paths": [str(path) for path in resumable_checkpoints],
            "batch_size": batch_size,
            "probe": probe,
            "peak_vram_bytes": max(
                record.get("peak_vram_bytes", 0) for record in probe
            ),
        })()
    else:
        neural = train_public_banglabert(
            safe_pool, validation, official, model_path=model_path,
            checkpoint_root=args.output_dir.parent / OUTPUT_NAME / "checkpoints",
        )
    validation_xc_prediction = (neural.validation_probability >= 0.5).astype(np.int64)
    official_xc_prediction = (neural.official_probability >= 0.5).astype(np.int64)
    validation_xc = _metric(validation_truth, validation_xc_prediction)
    official_xc = _metric(official_truth, official_xc_prediction, official_presence)

    v4a_oof_path = root / "artifacts" / "shared" / "v4a_0685" / "v4a_oof_probabilities.csv"
    if sha256_file(v4a_oof_path) != V4A_OOF_SHA256:
        raise RuntimeError("V13 V4-A OOF fallback authentication failed")
    v4a_oof = pd.read_csv(v4a_oof_path).sort_values("row_index")
    if v4a_oof.row_index.tolist() != list(range(len(official))) or not np.array_equal(v4a_oof.label, official_truth):
        raise RuntimeError("V13 V4-A OOF fallback alignment failed")
    v4a_prediction = (v4a_oof.probability_label1.to_numpy() >= 0.5).astype(np.int64)
    transferable = np.isin(official_xa_output.routes, TRANSFER_ROUTES)
    xd_prediction = v4a_prediction.copy()
    xd_prediction[transferable] = official_xa_output.predictions[transferable]
    official_xd = {
        **_metric(official_truth, xd_prediction, official_presence),
        "transfer": _transfer_metrics(official_truth, xd_prediction, official_xa_output.routes),
    }

    test_auth = authenticate_test_file(args.test_path)
    test = add_keys(validate_champion_test_frame(pd.read_csv(args.test_path)), workers=6)
    public_test_overlap = overlap_audit(public_pool, test)
    full_transfer = FrozenTemplateTransfer(public_pool)
    full_calibration = full_transfer.calibrate(validation)
    test_overlap = full_transfer.overlap_summary(test)
    transfer_validation_precision = validation_xa["transfer"]["precision"]
    gate_a = (
        official_xd["macro_f1"] >= 0.695
        and official_xd["transfer"]["coverage_count"] >= 30
        and (official_xd["transfer"]["precision"] or 0.0) >= 0.85
    )
    zero_conflicts = (
        public_audits["conflicts"]["train_pair"]["mixed_label_clusters"] == 0
        and public_audits["conflicts"]["contrastive_pair"]["mixed_label_clusters"] == 0
    )
    gate_b = (
        test_overlap["transfer_eligible_fraction"] >= 0.20
        and (transfer_validation_precision or 0.0) >= 0.95
        and zero_conflicts
    )
    xc_gate = (
        official_xc["macro_f1"] >= 0.70
        and official_xc["maximum_class_share"] <= 0.85
        and validation_xc["macro_f1"] >= 0.80
    )

    args.output_dir.mkdir(parents=True, exist_ok=resume_after_finalization_failure)
    args.submission_dir.mkdir(parents=True, exist_ok=False)
    generated: dict[str, Any] = {}
    if gate_a or gate_b:
        v4a_test_path = root / "artifacts" / "shared" / "v4a_0685" / "v4a_test_probabilities.csv"
        if sha256_file(v4a_test_path) != V4A_TEST_SHA256:
            raise RuntimeError("V13 V4-A test fallback authentication failed")
        v4a_test = pd.read_csv(v4a_test_path)
        if v4a_test.id.tolist() != test.id.tolist():
            raise RuntimeError("V13 V4-A test fallback alignment failed")
        fallback = (v4a_test.probability_label1.to_numpy() >= 0.5).astype(np.int64)
        transfer_output = full_transfer.predict(test)
        eligible = np.isin(transfer_output.routes, TRANSFER_ROUTES)
        fallback[eligible] = transfer_output.predictions[eligible]
        path = args.submission_dir / "submission_v13_transfer_fallback.csv"
        build_submission(test.id, fallback).to_csv(path, index=False)
        generated[path.name] = {
            "sha256": sha256_file(path),
            "prediction_counts": {str(label): int((fallback == label).sum()) for label in (0, 1)},
            "transfer_count": int(eligible.sum()),
        }
    if xc_gate:
        probability = infer_banglabert_test(
            test, model_path=model_path, checkpoint_paths=neural.checkpoint_paths,
            batch_size=neural.batch_size,
        )
        prediction = (probability >= 0.5).astype(np.int64)
        path = args.submission_dir / "submission_v13_public_banglabert.csv"
        build_submission(test.id, prediction).to_csv(path, index=False)
        generated[path.name] = {
            "sha256": sha256_file(path),
            "prediction_counts": {str(label): int((prediction == label).sum()) for label in (0, 1)},
        }

    result = {
        "experiment": "v13_public_template_transfer",
        "public_data": public_audits,
        "public_pool": pool_manifest(public_pool),
        "official_safe_pool": pool_manifest(safe_pool),
        "calibration": {"XA": calibration, "XA_full_test_index": full_calibration},
        "public_validation": {"XA": validation_xa, "XB": validation_xb, "XC": validation_xc},
        "official_external_diagnostic": {"XA": official_xa, "XB": official_xb, "XC": official_xc, "XD": official_xd},
        "selection": {"XD_gate_a": gate_a, "XD_gate_b": gate_b, "XC_gate": xc_gate, "generated": generated},
        "overlap": {"public_test": public_test_overlap, "test_transfer": test_overlap},
        "test_authentication": test_auth,
        "banglabert": {
            "revision": MODEL_REVISION,
            "batch_size": neural.batch_size,
            "probe": neural.probe,
            "seed_records": neural.seed_records,
            "peak_vram_bytes": neural.peak_vram_bytes,
        },
        "resources": {
            "runtime_seconds": perf_counter() - started,
            "workers": 6,
            "dataloader_workers": 4,
            "memory": memory_status(),
            "process_memory": process_memory_bytes(),
        },
        "safety": {
            "official_rows_used_for_fit": 0,
            "test_labels_accessed": False,
            "raw_test_text_printed_or_persisted": False,
            "leaderboard_used_for_rules": False,
            "manual_prediction_edits": False,
            "qwen_or_nli_run": False,
        },
    }
    _write_json(args.output_dir / "run_summary.json", result)
    _write_json(args.output_dir / "public_pool_manifest.json", {"public": pool_manifest(public_pool), "safe": pool_manifest(safe_pool), "audit": public_audits})
    _write_json(args.output_dir / "overlap_summary.json", result["overlap"])
    _write_json(args.submission_dir / "run_summary.json", result)
    _write_json(args.submission_dir / "public_pool_manifest.json", {"public": pool_manifest(public_pool), "safe": pool_manifest(safe_pool)})
    _write_json(args.submission_dir / "overlap_summary.json", result["overlap"])
    checksums = {path.name: sha256_file(path) for path in sorted(args.submission_dir.iterdir()) if path.is_file()}
    _write_json(args.submission_dir / "artifact_checksums.json", checksums)
    print(json.dumps({"status": "complete", "generated_submissions": sorted(generated), "runtime_seconds": result["resources"]["runtime_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
