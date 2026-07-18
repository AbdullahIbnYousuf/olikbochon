"""CLI for the single frozen V9 multilingual NLI verification experiment."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .data_loading import validate_labeled_frame
from .v4_preprocessing import official_context_is_present
from .v4_runner import load_official_training_frame
from .v4_validation import FOLD_COUNT, VALIDATION_SEEDS
from .v7_corpus import load_corpus, require_corpus_path
from .v7_retrieval import CharacterTfidfRetriever, RetrievedPassage
from .v9_features import (
    CANDIDATES,
    EXPECTED_ABSENT_ROWS,
    EXPECTED_PRESENT_ROWS,
    evaluate_candidates,
    retrieval_feature_matrix,
)
from .v9_nli import run_frozen_nli_inference


EXPERIMENT_NAME = "v9_frozen_nli_verification"
MODE = "frozen-nli-verification"
OUTPUT_NAME = "frozen_nli_verification"
EXPECTED_CORPUS_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
EXPECTED_CORPUS_CHUNKS = 301
EXPECTED_CORPUS_ARTICLES = 62_153
PRIMARY_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
CONTROL_REVISION = "b227ee8435ceadfa86dc1368a34254e2838bf242"
PRIMARY_WEIGHT_SHA256 = "7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541"
CONTROL_WEIGHT_SHA256 = "8869b0c99ad35ec8a8c92434b54383d2dfd7db8cd460e28b9944a407e3a423e4"
MAXIMUM_PROCESS_RAM_BYTES = 20 * 1024**3
MAXIMUM_SYSTEM_RAM_FRACTION = 0.80


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


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


def memory_status() -> dict[str, float | int]:
    system = _MemoryStatus()
    system.dwLength = ctypes.sizeof(system)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(system)):
        raise OSError("Unable to query V9 system memory")
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    get_process_memory.restype = ctypes.c_int
    handle = get_current_process()
    if not get_process_memory(
        handle, ctypes.byref(counters), counters.cb
    ):
        raise OSError("Unable to query V9 process memory")
    return {
        "system_total_physical_bytes": int(system.ullTotalPhys),
        "system_available_physical_bytes": int(system.ullAvailPhys),
        "system_memory_fraction": float(system.dwMemoryLoad / 100.0),
        "process_working_set_bytes": int(counters.WorkingSetSize),
        "process_peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def _parallel_retrieve(
    retriever: CharacterTfidfRetriever,
    queries: Sequence[str],
    *,
    workers: int,
) -> tuple[tuple[RetrievedPassage, ...], ...]:
    if workers != 4:
        raise ValueError("V9 corpus/retrieval preprocessing requires four CPU workers")

    def retrieve(query: str) -> tuple[RetrievedPassage, ...]:
        return retriever.retrieve(query, top_k=5, snippet_method="query_centered")

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v9-retrieval") as executor:
        rows = tuple(executor.map(retrieve, queries))
    if len(rows) != len(queries) or any(len(row) != 5 for row in rows):
        raise RuntimeError("V9 parallel retrieval did not return exactly top-5 evidence")
    return rows


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_exact_path(candidate: Path, expected: Path, description: str) -> Path:
    if not candidate.is_absolute():
        raise ValueError(f"{description} must be an absolute path")
    resolved = candidate.resolve()
    if resolved != expected.resolve():
        raise ValueError(f"{description} differs from the approved frozen path")
    return resolved


def validate_arguments(args: argparse.Namespace, root: Path) -> tuple[Path, Path, Path, Path]:
    if tuple(args.seeds) != VALIDATION_SEEDS or args.folds != FOLD_COUNT:
        raise ValueError("V9 requires seeds 17 29 43 and five folds")
    if args.cpu_workers != 4 or args.fold_workers != 3:
        raise ValueError("V9 requires four preprocessing workers and three fold workers")
    corpus = _require_exact_path(
        require_corpus_path(args.corpus_path), root / "data" / "retrieval" / "bnwiki", "corpus"
    )
    primary = _require_exact_path(
        args.primary_model_path,
        root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{PRIMARY_REVISION}",
        "primary model",
    )
    control = _require_exact_path(
        args.control_model_path,
        root / "data" / "models" / "v9_nli" / f"xlmr_large_xnli@{CONTROL_REVISION}",
        "control model",
    )
    output = _require_exact_path(
        args.output_dir, root / "artifacts" / "v9" / OUTPUT_NAME, "output directory"
    )
    if output.exists():
        raise ValueError("V9 output directory already exists")
    rules = set((root / ".gitignore").read_text(encoding="utf-8").splitlines())
    if "artifacts/v9/" not in rules:
        raise ValueError("artifacts/v9/ must be explicitly ignored")
    return corpus, primary, control, output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v9_runner",
        description="Run the single frozen official-only V9 multilingual NLI experiment.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--primary-model-path", required=True, type=Path)
    parser.add_argument("--control-model-path", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--cpu-workers", required=True, type=int)
    parser.add_argument("--fold-workers", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def frozen_config() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": MODE,
        "candidates": list(CANDIDATES),
        "routes": {"context_present": "frozen_exact_substring", "context_absent": "R/S/T"},
        "retrieval": {
            "top_k": 5,
            "query": "normalized_prompt",
            "snippet": "query_centered_800_normalized_characters",
            "cutoff": None,
            "corpus_manifest": EXPECTED_CORPUS_MANIFEST,
        },
        "nli": {
            "maximum_length": 384,
            "premise": "retrieved_evidence_snippet",
            "hypothesis": "response_bn",
            "truncation": "premise_first",
            "fine_tuned": False,
            "models_loaded_simultaneously": False,
        },
        "classifier": {
            "standard_scaler_train_partition_only": True,
            "class_weight": "balanced",
            "C": 1.0,
            "max_iter": 3000,
            "solver": "liblinear",
            "random_state": 42,
            "threshold": 0.5,
        },
        "parallelism": {
            "preprocessing_workers": 4,
            "nli_dataloader_workers": 2,
            "requested_fold_workers": 3,
            "start_method": "spawn",
            "blas_threads": 1,
        },
        "test_data_accessed": False,
        "public_labeled_data_used": False,
        "raw_text_persisted": False,
    }


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_parser().parse_args(argv)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"{variable} must equal 1")
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    root = repository_root()
    corpus_path, primary_path, control_path, output = validate_arguments(args, root)
    articles, corpus_manifest = load_corpus(corpus_path)
    if (
        corpus_manifest.logical_content_manifest_sha256 != EXPECTED_CORPUS_MANIFEST
        or corpus_manifest.unique_chunk_count != EXPECTED_CORPUS_CHUNKS
        or corpus_manifest.usable_article_count != EXPECTED_CORPUS_ARTICLES
    ):
        raise RuntimeError("V9 corpus authentication failed")
    memory_after_corpus = memory_status()
    retriever = CharacterTfidfRetriever(articles)
    memory_after_index = memory_status()
    frame = validate_labeled_frame(load_official_training_frame(root)).reset_index(drop=True)
    presence = np.asarray(
        [official_context_is_present(value) for value in frame["context"]], dtype=bool
    )
    if int(presence.sum()) != EXPECTED_PRESENT_ROWS or int((~presence).sum()) != EXPECTED_ABSENT_ROWS:
        raise RuntimeError("V9 corrected route totals differ from 130/169")
    null_frame = frame.loc[~presence].reset_index(drop=True)
    evidence = _parallel_retrieve(
        retriever, [str(value) for value in null_frame["prompt_bn"]], workers=args.cpu_workers
    )
    retrieval = retrieval_feature_matrix(null_frame, evidence)
    memory_after_retrieval = memory_status()
    print(json.dumps({"status": "retrieval_complete", "rows": len(evidence)}, sort_keys=True))
    primary = run_frozen_nli_inference(
        primary_path, evidence, null_frame["response_bn"].astype(str).tolist(), use_fast=True
    )
    if primary.identity.revision != PRIMARY_REVISION or primary.identity.weight_sha256 != PRIMARY_WEIGHT_SHA256:
        raise RuntimeError("V9 primary model identity changed")
    print(
        json.dumps(
            {
                "status": "primary_inference_complete",
                "revision": primary.identity.revision,
                "batch_size": primary.diagnostics["selected_batch_size"],
            },
            sort_keys=True,
        )
    )
    control = run_frozen_nli_inference(
        control_path, evidence, null_frame["response_bn"].astype(str).tolist(), use_fast=False
    )
    if control.identity.revision != CONTROL_REVISION or control.identity.weight_sha256 != CONTROL_WEIGHT_SHA256:
        raise RuntimeError("V9 control model identity changed")
    print(
        json.dumps(
            {
                "status": "control_inference_complete",
                "revision": control.identity.revision,
                "batch_size": control.diagnostics["selected_batch_size"],
            },
            sort_keys=True,
        )
    )
    before_classifiers = memory_status()
    actual_fold_workers = args.fold_workers
    if (
        before_classifiers["system_memory_fraction"] >= MAXIMUM_SYSTEM_RAM_FRACTION
        or before_classifiers["process_working_set_bytes"] >= MAXIMUM_PROCESS_RAM_BYTES
    ):
        actual_fold_workers = 1
    output.mkdir(parents=True, exist_ok=False)
    temporary_root = output / "fold_workers"
    temporary_root.mkdir()
    evaluation = evaluate_candidates(
        frame,
        primary.features,
        control.features,
        retrieval,
        fold_workers=actual_fold_workers,
        temporary_root=temporary_root,
    )
    shutil.rmtree(temporary_root)
    final_memory = memory_status()
    accepted = [
        candidate
        for candidate, record in evaluation["candidates"].items()
        if record["verdict"]["accepted"]
    ]
    selected = (
        max(
            accepted,
            key=lambda candidate: evaluation["candidates"][candidate]["routed"]["macro_f1"],
        )
        if accepted
        else None
    )
    result = {
        "experiment": EXPERIMENT_NAME,
        "status": "complete",
        "selection": {"selected_candidate": selected, "all_candidates_rejected": not accepted},
        "model_authentication": {
            "primary": {**primary.identity.__dict__, "diagnostics": primary.diagnostics},
            "control": {**control.identity.__dict__, "diagnostics": control.diagnostics},
        },
        "corpus_manifest": corpus_manifest.to_dict(),
        "retrieval_index": retriever.fit_audit,
        "evaluation": evaluation,
        "resources": {
            "cpu_preprocessing_workers": args.cpu_workers,
            "requested_fold_workers": args.fold_workers,
            "actual_fold_workers": actual_fold_workers,
            "memory_after_corpus": memory_after_corpus,
            "memory_after_index": memory_after_index,
            "memory_after_retrieval": memory_after_retrieval,
            "memory_before_classifiers": before_classifiers,
            "final_memory": final_memory,
            "runtime_seconds": perf_counter() - started,
        },
        "safety": {
            "official_rows": len(frame),
            "context_present_rows": int(presence.sum()),
            "context_absent_rows": int((~presence).sum()),
            "competition_test_opened": False,
            "public_labeled_data_used": False,
            "nli_fine_tuning_performed": False,
            "threshold_tuned": False,
            "raw_text_persisted": False,
            "row_level_probabilities_persisted": False,
            "output_inside_artifacts_v9": True,
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
                "selected_candidate": selected,
                "all_candidates_rejected": not accepted,
                "runtime_seconds": result["resources"]["runtime_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
