"""One-shot V12 deadline QA-verifier discovery, holdout, and gated inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .champion_submission import validate_champion_test_frame
from .champion_submission_runner import authenticate_test_file
from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from .metrics import classification_metrics
from .submission import build_submission
from .v10_ensemble import CANDIDATE_I, _label1_probability, _pipeline
from .v11_error_audit import blind_taxonomy, lock_discovery_split
from .v12_demo_retrieval import Demonstration, FrozenDemonstrationIndex, normalized_pair
from .v12_qwen_judge import JudgeOutput, OfflineQwenJudge, Target
from .v4_preprocessing import official_context_is_present
from .v5_lexical import build_v5_folds, build_v5_groups, deterministic_substring_prediction
from .v5_sparse import SparseNullModel
from .v7_corpus import load_corpus
from .v7_retrieval import CharacterTfidfRetriever
from .v9_features import retrieval_feature_matrix
from .v9_nli import run_frozen_nli_inference
from .v9_runner import _parallel_retrieve, memory_status


MODE = "deadline-qa-verifier"
OUTPUT_NAME = "deadline_qa_verifier"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_MANIFEST_SHA256 = "3e5901f633796ce01cddff36c8136a3d80095147e7a0126c66ce54d716ff1251"
PRIMARY_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
PRIMARY_WEIGHT_SHA256 = "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541"
CORPUS_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
PUBLIC_FILES = {
    "bangla_hallucination_5k_train.json": (4000, "0f988c5b09ba1b5214a93adc04e994a3c915fb0aff7eca9e2b1aca36aa62c07f"),
    "bangla_hallucination_5k_validation.json": (1000, "308c2e55bacc7552d6421101060a9b0e8c56136dd90b5cdc55f66f6375d3fa66"),
}
CONTRASTIVE_AUTH = (5000, "cbba66057253545d39914c147843bdca88a299baa0ea6ae6932a09d64e2e8d13")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the one frozen V12 deadline verifier.")
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--retrieval-workers", type=int, required=True)
    parser.add_argument("--similarity-workers", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    for file in sorted(value for value in path.iterdir() if value.is_file()):
        records.append({"name": file.name, "bytes": file.stat().st_size, "sha256": sha256_file(file)})
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return records, digest


def _load_public(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    audit: dict[str, Any] = {"used": [], "excluded_contrastive": None}
    for name, (count, digest) in PUBLIC_FILES.items():
        path = root / "data" / "public_5k" / name
        if sha256_file(path) != digest:
            raise RuntimeError(f"V12 public file authentication failed: {name}")
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        if len(frame) != count or tuple(sorted(frame.columns)) != (
            "context", "label", "prompt_bn", "response_bn"
        ):
            raise RuntimeError(f"V12 public file schema/count changed: {name}")
        if set(frame.label.unique()) != {0, 1}:
            raise RuntimeError(f"V12 public labels changed: {name}")
        frames.append(frame)
        audit["used"].append({"filename": name, "rows": count, "sha256": digest})
    contrastive = root / "data" / "public_5k" / "bangla_hallucination_5k_contrastive.json"
    contrastive_rows = len(json.loads(contrastive.read_text(encoding="utf-8")))
    if (contrastive_rows, sha256_file(contrastive)) != CONTRASTIVE_AUTH:
        raise RuntimeError("V12 contrastive artifact authentication changed")
    audit["excluded_contrastive"] = {
        "rows": contrastive_rows,
        "sha256": CONTRASTIVE_AUTH[1],
        "reason": "authenticated 5000 rows conflicts with historical 1000-row expectation; audit-only",
    }
    return pd.concat(frames, ignore_index=True), audit


def _demo_pool(
    public: pd.DataFrame,
    official: pd.DataFrame,
    official_indices: Sequence[int],
    group_ids: Sequence[str],
) -> list[Demonstration]:
    records = [
        Demonstration("public", int(index), None, str(row.prompt_bn), str(row.response_bn), int(row.label))
        for index, row in public.iterrows()
    ]
    records.extend(
        Demonstration(
            "official",
            int(index),
            str(group_ids[index]),
            str(official.iloc[index].prompt_bn),
            str(official.iloc[index].response_bn),
            int(official.iloc[index].label),
        )
        for index in official_indices
    )
    return records


def _targets(
    frame: pd.DataFrame,
    indices: Sequence[int],
    index: FrozenDemonstrationIndex,
    groups: Sequence[str],
) -> list[Target]:
    return [
        Target(
            int(position),
            int(position),
            str(frame.iloc[position].prompt_bn),
            str(frame.iloc[position].response_bn),
            index.retrieve(
                frame.iloc[position].prompt_bn,
                frame.iloc[position].response_bn,
                target_source="official",
                target_index=int(position),
                excluded_group=str(groups[position]),
            ),
        )
        for position in indices
    ]


def _metrics(truth: np.ndarray, outputs: Sequence[JudgeOutput]) -> dict[str, Any]:
    prediction = np.asarray([value.label for value in outputs], dtype=np.int64)
    counts = np.bincount(prediction, minlength=2)
    override = np.asarray([value.exact_template_override for value in outputs], dtype=bool)
    invalid = sum(value.invalid_outputs for value in outputs)
    return {
        **classification_metrics(truth, prediction),
        "correct_count": int(np.sum(prediction == truth)),
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "maximum_class_share": float(counts.max() / len(prediction)),
        "exact_template_coverage": float(override.mean()),
        "exact_template_count": int(override.sum()),
        "exact_template_accuracy": float(np.mean(prediction[override] == truth[override])) if override.any() else None,
        "non_template_accuracy": float(np.mean(prediction[~override] == truth[~override])) if (~override).any() else None,
        "invalid_json_count": int(invalid),
        "invalid_output_rate": float(invalid / len(prediction)),
    }


def _subgroups(
    truth: np.ndarray, outputs: Sequence[JudgeOutput], taxonomy: pd.DataFrame
) -> dict[str, Any]:
    predictions = np.asarray([value.label for value in outputs], dtype=np.int64)
    records: dict[str, Any] = {}
    for family, block in taxonomy.groupby("primary_task_family", sort=True):
        positions = block.index.to_numpy(dtype=np.int64)
        records[str(family)] = {"support": len(positions), **classification_metrics(truth[positions], predictions[positions])}
    tags = taxonomy.claim_risk_tags
    for name, needle in (
        ("numeric_or_date", ("numeric claim", "date or temporal claim")),
        ("factual_entity", None),
        ("generic_answer", ("generic low-information answer",)),
    ):
        mask = (
            taxonomy.primary_task_family.eq("factual entity question").to_numpy()
            if needle is None
            else tags.map(lambda values, wanted=needle: any(tag in values for tag in wanted)).to_numpy()
        )
        records[name] = {
            "support": int(mask.sum()),
            **classification_metrics(truth[mask], predictions[mask]),
        } if mask.any() else {"support": 0}
    return records


def _baseline_oof(
    frame: pd.DataFrame, target_indices: Sequence[int], candidate_r: np.ndarray
) -> dict[str, Any]:
    presence = np.asarray([official_context_is_present(value) for value in frame.context], dtype=bool)
    absent_indices = np.flatnonzero(~presence)
    position = {int(index): offset for offset, index in enumerate(absent_indices)}
    targets = set(map(int, target_indices))
    folds = build_v5_folds(frame)
    groups = np.asarray(folds.audit.group_ids, dtype=object)
    probabilities = {name: {index: [] for index in targets} for name in ("I", "R")}
    for split in folds.folds:
        train = np.asarray(split.train_indices, dtype=np.int64)
        validation = np.asarray([index for index in split.validation_indices if index in targets])
        if validation.size == 0:
            continue
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("V12 baseline OOF group overlap")
        train = train[~presence[train]]
        train_positions = np.asarray([position[int(index)] for index in train])
        validation_positions = np.asarray([position[int(index)] for index in validation])
        train_frame = frame.iloc[train].reset_index(drop=True)
        validation_frame = frame.iloc[validation].reset_index(drop=True)
        model_i = SparseNullModel(CANDIDATE_I).fit(train_frame)
        values_i = model_i.predict_label1_probability(validation_frame)
        model_r = _pipeline().fit(candidate_r[train_positions], train_frame.label.to_numpy())
        values_r = _label1_probability(model_r, candidate_r[validation_positions])
        for index, value_i, value_r in zip(validation, values_i, values_r, strict=True):
            probabilities["I"][int(index)].append(float(value_i))
            probabilities["R"][int(index)].append(float(value_r))
    truth = frame.iloc[list(target_indices)].label.to_numpy(dtype=np.int64)
    result = {}
    for name in ("I", "R"):
        values = np.asarray([np.mean(probabilities[name][int(index)]) for index in target_indices])
        if any(len(probabilities[name][int(index)]) != 3 for index in target_indices):
            raise RuntimeError("V12 holdout baseline lacks exact three-seed OOF coverage")
        result[name] = classification_metrics(truth, (values >= 0.5).astype(np.int64))
    return result


def _candidate_r_matrix(root: Path, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    presence = np.asarray([official_context_is_present(value) for value in frame.context], dtype=bool)
    null_frame = frame.loc[~presence].reset_index(drop=True)
    articles, manifest = load_corpus(root / "data" / "retrieval" / "bnwiki")
    if manifest.logical_content_manifest_sha256 != CORPUS_MANIFEST:
        raise RuntimeError("V12 Candidate R corpus authentication failed")
    retriever = CharacterTfidfRetriever(articles)
    evidence = _parallel_retrieve(retriever, null_frame.prompt_bn.astype(str).tolist(), workers=4)
    retrieval = retrieval_feature_matrix(null_frame, evidence)
    nli = run_frozen_nli_inference(
        root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{PRIMARY_REVISION}",
        evidence,
        null_frame.response_bn.astype(str).tolist(),
        use_fast=True,
    )
    if nli.identity.revision != PRIMARY_REVISION or nli.identity.weight_sha256 != PRIMARY_WEIGHT_SHA256:
        raise RuntimeError("V12 Candidate R authentication failed")
    return np.hstack((nli.features, retrieval)), {**nli.identity.__dict__, "diagnostics": nli.diagnostics}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_parser().parse_args(argv)
    if args.retrieval_workers != 4 or args.similarity_workers != 4:
        raise ValueError("V12 requires four retrieval and four similarity workers")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"{variable} must equal 1")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    root = repository_root()
    expected_output = root / "artifacts" / "v12" / OUTPUT_NAME
    if args.output_dir.resolve() != expected_output.resolve() or args.output_dir.exists():
        raise ValueError("V12 output path changed or already exists")
    model_path = root / "data" / "models" / "v12_qwen3" / f"qwen3-4b@{MODEL_REVISION}"
    model_files, model_manifest = _sha_manifest(model_path)
    if model_manifest != MODEL_MANIFEST_SHA256:
        raise RuntimeError("V12 Qwen snapshot manifest changed")
    frame = load_labeled_json(
        root / "data" / "competition" / "dataset samples.json",
        expected_sha256=OFFICIAL_SAMPLE_SHA256,
    ).reset_index(drop=True)
    locked = lock_discovery_split(frame, 0.75)
    if locked.record["discovery_row_index_sha256"] != "775d9a0170fa4e5aacceb1815243ed58cb39beced475b0269ffb911872146320" or locked.record["holdout_row_index_sha256"] != "c0ce983882e73778dc0de857c8ed708c518111318c03ef998d739289124e42db":
        raise RuntimeError("V12 locked split hashes changed")
    groups = build_v5_groups(frame).group_ids
    public, public_audit = _load_public(root)
    index = FrozenDemonstrationIndex(_demo_pool(public, frame, locked.discovery_indices, groups))
    discovery_targets = _targets(frame, locked.discovery_indices, index, groups)
    taxonomy = blind_taxonomy(frame.iloc[locked.discovery_indices], workers=4)
    truth = frame.iloc[locked.discovery_indices].label.to_numpy(dtype=np.int64)
    judge = OfflineQwenJudge(str(model_path))
    aa_outputs, aa_batch = judge.judge_aa(discovery_targets, start_batch_size=4)
    aa = {**_metrics(truth, aa_outputs), "subgroups": _subgroups(truth, aa_outputs, taxonomy)}
    ab_outputs = judge.judge_ab(discovery_targets)
    ab = {**_metrics(truth, ab_outputs), "subgroups": _subgroups(truth, ab_outputs, taxonomy)}
    acceptable = {
        "AA": aa["maximum_class_share"] <= 0.85 and aa["invalid_output_rate"] <= 0.02,
        "AB": ab["maximum_class_share"] <= 0.85 and ab["invalid_output_rate"] <= 0.02,
    }
    if not any(acceptable.values()):
        winner = None
    elif acceptable["AA"] and (not acceptable["AB"] or aa["macro_f1"] >= ab["macro_f1"] - 0.01):
        winner = "AA"
    else:
        winner = "AB"
    if winner is None:
        raise RuntimeError("V12 both discovery candidates failed frozen safety selection")
    frozen_winner = winner
    holdout_targets = _targets(frame, locked.holdout_indices, index, groups)
    holdout_outputs = (
        judge.judge_aa(holdout_targets, start_batch_size=aa_batch)[0]
        if frozen_winner == "AA"
        else judge.judge_ab(holdout_targets)
    )
    holdout_truth = frame.iloc[locked.holdout_indices].label.to_numpy(dtype=np.int64)
    holdout = _metrics(holdout_truth, holdout_outputs)
    qwen_peak = judge.peak_vram
    judge.close()
    del judge
    gc.collect()
    candidate_r, r_identity = _candidate_r_matrix(root, frame)
    baselines = _baseline_oof(frame, locked.holdout_indices, candidate_r)
    del candidate_r
    gc.collect()
    torch.cuda.empty_cache()
    stronger = max(baselines["I"]["macro_f1"], baselines["R"]["macro_f1"])
    holdout["improvement_over_stronger_baseline"] = float(holdout["macro_f1"] - stronger)
    gate_passed = (
        holdout["macro_f1"] >= 0.70
        and holdout["correct_count"] >= 30
        and holdout["macro_f1"] > baselines["I"]["macro_f1"]
        and holdout["macro_f1"] > baselines["R"]["macro_f1"]
        and holdout["maximum_class_share"] <= 0.85
        and holdout["invalid_output_rate"] == 0.0
    )
    test_path = root / "data" / "competition" / "test set.csv"
    test_auth = authenticate_test_file(test_path)
    test = validate_champion_test_frame(pd.read_csv(test_path))
    full_index = FrozenDemonstrationIndex(
        _demo_pool(public, frame, np.flatnonzero([not official_context_is_present(v) for v in frame.context]), groups)
    )
    public_index = FrozenDemonstrationIndex(_demo_pool(public, frame, (), groups))
    overlap = public_index.aggregate_overlap(test.prompt_bn.tolist(), test.response_bn.tolist())
    submission_record: dict[str, Any] = {"generated": False}
    if gate_passed:
        presence = np.asarray([official_context_is_present(value) for value in test.context], dtype=bool)
        absent_positions = np.flatnonzero(~presence)
        test_targets = [
            Target(None, test.iloc[position].id, str(test.iloc[position].prompt_bn), str(test.iloc[position].response_bn), full_index.retrieve(test.iloc[position].prompt_bn, test.iloc[position].response_bn))
            for position in absent_positions
        ]
        judge = OfflineQwenJudge(str(model_path))
        test_outputs = judge.judge_aa(test_targets, start_batch_size=aa_batch)[0] if frozen_winner == "AA" else judge.judge_ab(test_targets)
        qwen_peak = max(qwen_peak, judge.peak_vram)
        judge.close()
        predictions = np.empty(len(test), dtype=np.int64)
        predictions[presence] = np.asarray([
            deterministic_substring_prediction(row.context, row.response_bn)
            for row in test.loc[presence].itertuples(index=False)
        ])
        predictions[~presence] = np.asarray([value.label for value in test_outputs])
        submission_dir = root / "artifacts" / "submissions" / "v12_deadline_qa_verifier"
        submission_dir.mkdir(parents=True, exist_ok=False)
        submission = build_submission(test.id, predictions)
        submission.to_csv(submission_dir / "submission.csv", index=False)
        output_by_position = {position: value for position, value in zip(absent_positions, test_outputs, strict=True)}
        scores = pd.DataFrame({
            "id": test.id,
            "route": np.where(presence, "context_present", "context_absent"),
            "exact_template_override": [False if presence[i] else output_by_position[i].exact_template_override for i in range(len(test))],
            "qwen_label": [pd.NA if presence[i] else output_by_position[i].label for i in range(len(test))],
            "qwen_confidence": [pd.NA if presence[i] else output_by_position[i].confidence for i in range(len(test))],
            "predicted_label": predictions,
        })
        scores.to_csv(submission_dir / "v12_scores.csv", index=False)
        primary_hashes = {name: sha256_file(submission_dir / name) for name in ("submission.csv", "v12_scores.csv")}
        submission_record = {
            "generated": True,
            "directory": str(submission_dir),
            "route_counts": {"context_present": int(presence.sum()), "context_absent": int((~presence).sum())},
            "predicted_label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(predictions, minlength=2))},
            "hashes": primary_hashes,
        }
    elif (
        overlap["exact_prompt_response_fraction"] > 0.20
        and overlap["exact_matches_label_consistent"]
    ):
        v4a_path = root / "artifacts" / "shared" / "v4a_0685" / "v4a_test_probabilities.csv"
        if sha256_file(v4a_path) != "2dd5ffa97c285727143fefe7d172bd02588368b7c3ff111088709ce0eb27adc2":
            raise RuntimeError("V12 emergency V4-A fallback authentication failed")
        v4a = pd.read_csv(v4a_path)
        if tuple(v4a.columns) != ("id", "probability_label1") or v4a.id.tolist() != test.id.tolist():
            raise RuntimeError("V12 emergency V4-A fallback alignment failed")
        predictions = (v4a.probability_label1.to_numpy(dtype=np.float64) >= 0.50).astype(np.int64)
        overrides = 0
        for position, row in enumerate(test.itertuples(index=False)):
            labels = public_index.exact_labels.get(normalized_pair(row.prompt_bn, row.response_bn), set())
            if len(labels) == 1:
                predictions[position] = next(iter(labels))
                overrides += 1
        submission_dir = root / "artifacts" / "submissions" / "v12_deadline_qa_verifier"
        submission_dir.mkdir(parents=True, exist_ok=False)
        fallback = submission_dir / "submission_exact_template_fallback.csv"
        build_submission(test.id, predictions).to_csv(fallback, index=False)
        submission_record = {
            "generated": False,
            "emergency_template_fallback_generated": True,
            "emergency_exact_override_count": overrides,
            "emergency_path": str(fallback),
            "emergency_sha256": sha256_file(fallback),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment": "v12_deadline_qa_verifier",
        "public_data": public_audit,
        "model": {"repo": "Qwen/Qwen3-4B", "revision": MODEL_REVISION, "license": "Apache-2.0", "manifest_sha256": model_manifest, "files": model_files},
        "split": locked.record,
        "retrieval": {"discovery_pool_rows": len(index.demonstrations), "discovery_pool_manifest": index.manifest_sha256, "full_pool_rows": len(full_index.demonstrations), "full_pool_manifest": full_index.manifest_sha256},
        "discovery": {"AA": aa, "AB": ab, "selected": frozen_winner, "AA_final_batch_size": aa_batch},
        "holdout": {"winner": frozen_winner, "metrics": holdout, "candidate_baselines": baselines, "gate_passed": gate_passed, "high_potential": holdout["macro_f1"] >= 0.80 and holdout["correct_count"] >= 34},
        "test_structural_overlap": overlap,
        "test_authentication": test_auth,
        "submission": submission_record,
        "candidate_r_authentication": r_identity,
        "resources": {"runtime_seconds": perf_counter() - started, "peak_qwen_vram_bytes": qwen_peak, "retrieval_workers": 4, "similarity_workers": 4, "memory": memory_status()},
        "safety": {"holdout_used_for_selection": False, "holdout_manual_review": False, "raw_test_text_printed_or_persisted": False, "fine_tuning": False, "external_inference_api": False},
    }
    _write_json(args.output_dir / "run_summary.json", result)
    if submission_record["generated"]:
        submission_dir = Path(submission_record["directory"])
        _write_json(submission_dir / "run_summary.json", result)
        checksums = {file.name: sha256_file(file) for file in sorted(submission_dir.iterdir()) if file.is_file()}
        _write_json(submission_dir / "artifact_checksums.json", checksums)
    print(json.dumps({"status": "complete", "winner": frozen_winner, "holdout_gate_passed": gate_passed, "submission_generated": submission_record["generated"], "runtime_seconds": result["resources"]["runtime_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
