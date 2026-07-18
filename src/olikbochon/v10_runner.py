"""CLI for the single frozen V10 nested lexical-semantic ensemble experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .data_loading import validate_labeled_frame
from .v10_ensemble import run_nested_ensemble
from .v4_preprocessing import official_context_is_present
from .v4_runner import load_official_training_frame
from .v4_validation import FOLD_COUNT, VALIDATION_SEEDS
from .v7_corpus import load_corpus, require_corpus_path
from .v7_retrieval import CharacterTfidfRetriever
from .v9_features import retrieval_feature_matrix
from .v9_nli import run_frozen_nli_inference
from .v9_runner import _parallel_retrieve, memory_status


EXPERIMENT_NAME = "v10_nested_lexical_semantic_ensemble"
MODE = "nested-lexical-semantic-ensemble"
OUTPUT_NAME = "nested_lexical_semantic_ensemble"
EXPECTED_CORPUS_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
EXPECTED_CORPUS_CHUNKS = 301
EXPECTED_CORPUS_ARTICLES = 62_153
PRIMARY_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
PRIMARY_WEIGHT_SHA256 = "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541"
EXPECTED_PRESENT_ROWS = 130
EXPECTED_ABSENT_ROWS = 169


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_exact_path(candidate: Path, expected: Path, description: str) -> Path:
    if not candidate.is_absolute():
        raise ValueError(f"V10 {description} must be an absolute path")
    resolved = candidate.resolve()
    if resolved != expected.resolve():
        raise ValueError(f"V10 {description} differs from the approved frozen path")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v10_runner",
        description="Run the frozen V10 nested Candidate I/R ensembles.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--primary-model-path", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--cpu-workers", required=True, type=int)
    parser.add_argument("--outer-workers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_arguments(args: argparse.Namespace, root: Path) -> tuple[Path, Path, Path]:
    if tuple(args.seeds) != VALIDATION_SEEDS or args.folds != FOLD_COUNT:
        raise ValueError("V10 requires seeds 17 29 43 and five outer folds")
    if args.cpu_workers != 4 or args.outer_workers != 3:
        raise ValueError("V10 requires four preprocessing workers and three outer workers")
    corpus = _require_exact_path(
        require_corpus_path(args.corpus_path), root / "data" / "retrieval" / "bnwiki", "corpus"
    )
    primary = _require_exact_path(
        args.primary_model_path,
        root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{PRIMARY_REVISION}",
        "primary model",
    )
    output = _require_exact_path(
        args.output_dir, root / "artifacts" / "v10" / OUTPUT_NAME, "output directory"
    )
    if output.exists():
        raise ValueError("V10 output directory already exists")
    rules = set((root / ".gitignore").read_text(encoding="utf-8").splitlines())
    if "artifacts/v10/" not in rules:
        raise ValueError("artifacts/v10/ must be explicitly ignored")
    return corpus, primary, output


def frozen_config() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "candidates": {
            "U": "fixed 0.50 Candidate I plus 0.50 Candidate R probability average",
            "V": "nested Candidate-I weight grid [0, 0.25, 0.50, 0.75, 1]",
            "W": "nested standardized logistic stacker on Candidate I/R probabilities",
        },
        "base_models": {
            "candidate_i": "frozen V5 candidate_i_sparse_lexical_union",
            "candidate_r": "frozen V9 mDeBERTa plus Candidate N retrieval features",
            "mdeberta_revision": PRIMARY_REVISION,
        },
        "decision_threshold": 0.50,
        "outer_validation": {"seeds": list(VALIDATION_SEEDS), "folds": 5},
        "inner_validation": {
            "folds": 3,
            "scope": "current outer training null-route rows only",
            "genuine_oof_base_probabilities": True,
        },
        "eligibility": {
            "minimum_routed_macro_f1": 0.697051,
            "minimum_improved_seeds": 2,
            "maximum_seed_standard_deviation": 0.04,
            "maximum_predicted_class_share": 0.90,
        },
        "parallelism": {
            "corpus_preprocessing_workers": 4,
            "nli_dataloader_workers": 2,
            "outer_workers": 3,
            "start_method": "spawn",
            "blas_threads": 1,
        },
        "competition_test_accessed": False,
        "leaderboard_used_for_fitting": False,
        "raw_text_persisted": False,
    }


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_parser().parse_args(argv)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"{variable} must equal 1")
    os.environ.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    )
    root = repository_root()
    corpus_path, primary_path, output = validate_arguments(args, root)
    articles, corpus_manifest = load_corpus(corpus_path)
    if (
        corpus_manifest.logical_content_manifest_sha256 != EXPECTED_CORPUS_MANIFEST
        or corpus_manifest.unique_chunk_count != EXPECTED_CORPUS_CHUNKS
        or corpus_manifest.usable_article_count != EXPECTED_CORPUS_ARTICLES
    ):
        raise RuntimeError("V10 corpus authentication failed")
    after_corpus = memory_status()
    retriever = CharacterTfidfRetriever(articles)
    after_index = memory_status()
    frame = validate_labeled_frame(load_official_training_frame(root)).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V10 corrected route totals differ from 130/169")
    null_frame = frame.loc[~presence].reset_index(drop=True)
    evidence = _parallel_retrieve(
        retriever, [str(value) for value in null_frame["prompt_bn"]], workers=args.cpu_workers
    )
    retrieval = retrieval_feature_matrix(null_frame, evidence)
    after_retrieval = memory_status()
    nli = run_frozen_nli_inference(
        primary_path, evidence, null_frame["response_bn"].astype(str).tolist(), use_fast=True
    )
    if nli.identity.revision != PRIMARY_REVISION or nli.identity.weight_sha256 != PRIMARY_WEIGHT_SHA256:
        raise RuntimeError("V10 mDeBERTa identity changed")
    candidate_r = np.hstack((nli.features, retrieval))
    before_outer = memory_status()
    if before_outer["system_memory_fraction"] >= 0.80:
        raise MemoryError("V10 cannot preserve three outer workers above 80% system RAM")
    output.mkdir(parents=True, exist_ok=False)
    temporary_root = output / "outer_workers"
    temporary_root.mkdir()
    evaluation = run_nested_ensemble(
        frame,
        candidate_r,
        fold_workers=args.outer_workers,
        temporary_root=temporary_root,
    )
    shutil.rmtree(temporary_root)
    final_memory = memory_status()
    result = {
        "experiment": EXPERIMENT_NAME,
        "status": "complete",
        "evaluation": evaluation,
        "model_authentication": {**nli.identity.__dict__, "diagnostics": nli.diagnostics},
        "corpus_manifest": corpus_manifest.to_dict(),
        "retrieval_index": retriever.fit_audit,
        "resources": {
            "cpu_preprocessing_workers": args.cpu_workers,
            "outer_workers": args.outer_workers,
            "memory_after_corpus": after_corpus,
            "memory_after_index": after_index,
            "memory_after_retrieval": after_retrieval,
            "memory_before_outer_workers": before_outer,
            "final_memory": final_memory,
            "runtime_seconds": perf_counter() - started,
        },
        "safety": {
            "official_rows": len(frame),
            "context_present_rows": int(presence.sum()),
            "context_absent_rows": int((~presence).sum()),
            "competition_test_opened": False,
            "leaderboard_used_for_fitting": False,
            "threshold_tuned": False,
            "additional_weights_tuned": False,
            "raw_text_persisted": False,
            "row_level_probabilities_persisted": False,
        },
    }
    (output / "frozen_config.json").write_text(
        json.dumps(frozen_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_candidate": evaluation["selection"]["selected_candidate"],
                "all_candidates_rejected": evaluation["selection"]["all_candidates_rejected"],
                "runtime_seconds": result["resources"]["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
