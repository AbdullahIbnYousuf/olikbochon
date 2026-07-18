"""CLI for the single discovery-only V11 label-definition error audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .data_loading import validate_labeled_frame
from .v11_error_audit import (
    analyze_discovery,
    blind_taxonomy,
    frozen_hypotheses,
    generate_discovery_oof,
    lock_discovery_split,
)
from .v4_runner import load_official_training_frame
from .v4_preprocessing import official_context_is_present
from .v7_corpus import load_corpus
from .v7_retrieval import CharacterTfidfRetriever
from .v9_features import retrieval_feature_matrix
from .v9_nli import run_frozen_nli_inference
from .v9_runner import _parallel_retrieve, memory_status


MODE = "label-definition-error-audit"
OUTPUT_NAME = "label_definition_error_audit"
EXPECTED_CORPUS_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
PRIMARY_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
PRIMARY_WEIGHT_SHA256 = "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the locked V11 discovery-only audit.")
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--discovery-fraction", required=True, type=float)
    parser.add_argument("--tag-workers", required=True, type=int)
    parser.add_argument("--similarity-workers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


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
    expected_output = root / "artifacts" / "v11" / OUTPUT_NAME
    if not args.output_dir.is_absolute() or args.output_dir.resolve() != expected_output.resolve():
        raise ValueError("V11 output directory differs from the approved ignored path")
    if args.output_dir.exists():
        raise ValueError("V11 output directory already exists")
    if "artifacts/v11/" not in (root / ".gitignore").read_text(encoding="utf-8").splitlines():
        raise ValueError("artifacts/v11/ must be explicitly ignored")

    frame = validate_labeled_frame(load_official_training_frame(root)).reset_index(drop=True)
    locked = lock_discovery_split(frame, args.discovery_fraction)
    discovery_frame = frame.iloc[locked.discovery_indices]
    taxonomy = blind_taxonomy(discovery_frame, workers=args.tag_workers)
    taxonomy_hash = taxonomy.attrs["taxonomy_sha256"]

    corpus_path = root / "data" / "retrieval" / "bnwiki"
    model_path = root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{PRIMARY_REVISION}"
    articles, corpus_manifest = load_corpus(corpus_path)
    if corpus_manifest.logical_content_manifest_sha256 != EXPECTED_CORPUS_MANIFEST:
        raise RuntimeError("V11 corpus authentication failed")
    retriever = CharacterTfidfRetriever(articles)
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]],
        dtype=bool,
    )
    null_frame = frame.loc[~presence].reset_index(drop=True)
    evidence = _parallel_retrieve(
        retriever, [str(value) for value in null_frame["prompt_bn"]], workers=4
    )
    retrieval = retrieval_feature_matrix(null_frame, evidence)
    nli = run_frozen_nli_inference(
        model_path, evidence, null_frame["response_bn"].astype(str).tolist(), use_fast=True
    )
    if nli.identity.revision != PRIMARY_REVISION or nli.identity.weight_sha256 != PRIMARY_WEIGHT_SHA256:
        raise RuntimeError("V11 frozen Candidate R model authentication failed")
    candidate_r_matrix = np.hstack((nli.features, retrieval))
    oof, provenance = generate_discovery_oof(
        frame, candidate_r_matrix, locked.discovery_indices
    )
    analysis, row_table = analyze_discovery(
        frame,
        locked.discovery_indices,
        taxonomy,
        oof,
        similarity_workers=args.similarity_workers,
    )
    hypotheses = frozen_hypotheses(analysis)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "experiment": "v11_label_definition_error_audit",
        "status": "complete",
        "locked_split": locked.record,
        "taxonomy": {
            "blind_pass_completed_before_outcome_join": True,
            "taxonomy_sha256": taxonomy_hash,
            "subjective_operational_definitions": {
                "ambiguous_or_underspecified": "normalized prompt shorter than 12 characters or matching the frozen generic-query lexicon",
                "unverifiable_specificity": "response over 160 normalized characters with a named-entity-like or numeric claim",
                "generic_low_information": "normalized response shorter than 25 characters",
            },
        },
        "oof_provenance": provenance,
        "analysis": analysis,
        "label_definition": {
            "label_0": "Operationally marks hallucinated or unacceptable responses, including factual, relevance, and instruction-following failures; it is not a pure truth-value class.",
            "label_1": "Operationally marks acceptable/faithful responses, including some generic, uncertain, creative, or subjective answers when appropriate to the request.",
            "caution": "Discovery patterns can be ambiguous or potentially inconsistent and require organizer clarification; V11 does not assert annotation errors.",
        },
        "frozen_v12_hypotheses": hypotheses,
        "resources": {
            "runtime_seconds": perf_counter() - started,
            "tag_workers": args.tag_workers,
            "similarity_workers": args.similarity_workers,
            "nested_process_pools": False,
            "memory": memory_status(),
            "nli_batch_size": nli.diagnostics["selected_batch_size"],
        },
        "safety": {
            "official_rows": len(frame),
            "competition_test_opened": False,
            "holdout_text_inspected_or_displayed": False,
            "holdout_metrics_computed": False,
            "holdout_predictions_generated": 0,
            "labels_changed": False,
            "new_candidate_trained_or_evaluated": False,
            "hypotheses_evaluated": False,
            "raw_text_persisted": False,
        },
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    row_table.to_csv(args.output_dir / "discovery_audit.csv", index=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "discovery_rows": locked.record["discovery_row_count"],
                "holdout_rows": locked.record["holdout_row_count"],
                "runtime_seconds": result["resources"]["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
