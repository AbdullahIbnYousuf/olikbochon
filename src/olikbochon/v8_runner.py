"""Constrained CLI for the authenticated V8 route-complement experiment."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .v4_preprocessing import official_context_is_present
from .v4_runner import load_official_training_frame
from .v4_validation import FOLD_COUNT, VALIDATION_SEEDS
from .v7_corpus import load_corpus
from .v7_retrieval import CharacterTfidfRetriever
from .v8_route_complement import (
    EXPERIMENT_NAME,
    build_v4a_feature_matrix,
    frozen_v8_config,
    run_authenticated_oof,
)


MODE = "authenticated-route-complement"
OUTPUT_NAME = "authenticated_route_complement"
EXPECTED_LOGICAL_MANIFEST = "af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf"
EXPECTED_UNIQUE_CHUNKS = 301
EXPECTED_USABLE_ARTICLES = 62_153
MAXIMUM_PROCESS_RAM_BYTES = 24 * 1024**3


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
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    current_process.restype = ctypes.c_void_p
    memory_info.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    memory_info.restype = ctypes.c_int
    if not memory_info(current_process(), ctypes.byref(counters), counters.cb):
        raise OSError("Unable to query V8 process memory")
    result = {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }
    if max(result.values()) > MAXIMUM_PROCESS_RAM_BYTES:
        raise MemoryError("V8 process RAM exceeded the frozen 24 GB safety limit")
    return result


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.v8_runner",
        description="Run the one authenticated V8 shared-fold route complement.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--corpus-path", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--folds", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_arguments(
    args: argparse.Namespace, root: Path
) -> tuple[Path, Path]:
    root = Path(root).resolve()
    expected_corpus = (root / "data" / "retrieval" / "bnwiki").resolve()
    expected_output = (root / "artifacts" / "v8" / OUTPUT_NAME).resolve()
    if not args.corpus_path.is_absolute() or not args.output_dir.is_absolute():
        raise ValueError("V8 corpus and output paths must be absolute")
    corpus = args.corpus_path.resolve()
    output = args.output_dir.resolve()
    if corpus != expected_corpus or not corpus.is_dir():
        raise ValueError("V8 corpus must be the approved data/retrieval/bnwiki directory")
    if tuple(args.seeds) != VALIDATION_SEEDS or args.folds != FOLD_COUNT:
        raise ValueError("V8 requires seeds 17 29 43 and five folds")
    if output != expected_output:
        raise ValueError(f"V8 output must be artifacts/v8/{OUTPUT_NAME}")
    ignore_lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    if "artifacts/v8/" not in ignore_lines:
        raise ValueError("artifacts/v8/ must be explicitly ignored")
    if output.exists():
        raise ValueError("V8 output already exists; overwrite is prohibited")
    return corpus, output


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_cli_parser().parse_args(argv)
    root = repository_root()
    corpus_path, output = validate_arguments(args, root)
    corpus_started = perf_counter()
    articles, manifest = load_corpus(corpus_path)
    corpus_seconds = perf_counter() - corpus_started
    if manifest.logical_content_manifest_sha256 != EXPECTED_LOGICAL_MANIFEST:
        raise RuntimeError("V8 corpus logical manifest changed")
    if manifest.unique_chunk_count != EXPECTED_UNIQUE_CHUNKS:
        raise RuntimeError("V8 corpus unique chunk count changed")
    if manifest.usable_article_count != EXPECTED_USABLE_ARTICLES:
        raise RuntimeError("V8 corpus usable article count changed")
    print(
        json.dumps(
            {
                "status": "corpus_authenticated",
                "logical_manifest": manifest.logical_content_manifest_sha256,
                "unique_chunks": manifest.unique_chunk_count,
                "usable_articles": manifest.usable_article_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    retriever = CharacterTfidfRetriever(articles)
    memory_after_index = process_memory_bytes()
    print(
        json.dumps(
            {
                "status": "retrieval_index_built",
                "index_rows": retriever.fit_audit["index_rows"],
                "index_columns": retriever.fit_audit["index_columns"],
                "official_queries_used_for_fit": 0,
                "peak_process_ram_bytes": memory_after_index[
                    "peak_working_set_bytes"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frame = load_official_training_frame(root)
    presence = frame["context"].map(official_context_is_present)
    absent = frame.loc[~presence].reset_index(drop=True)
    evidence = retriever.retrieve_queries(
        [str(value) for value in absent["prompt_bn"]],
        top_k=1,
        snippet_method="intro",
    )
    features = build_v4a_feature_matrix(frame, evidence)
    results, provenance = run_authenticated_oof(
        frame,
        features,
        corpus_manifest_sha256=manifest.logical_content_manifest_sha256,
    )
    final_memory = process_memory_bytes()
    results["runtime"] = {
        "total_seconds": perf_counter() - started,
        "corpus_load_seconds": corpus_seconds,
        "index_build_seconds": retriever.fit_audit["index_build_seconds"],
    }
    results["process_memory"] = final_memory
    results["corpus_manifest"] = manifest.to_dict()
    results["retrieval_index"] = retriever.fit_audit
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "frozen_config.json", frozen_v8_config())
    _write_json(output / "oof_provenance.json", provenance)
    _write_json(output / "results.json", results)
    print(
        json.dumps(
            {
                "status": "complete",
                "experiment": EXPERIMENT_NAME,
                "v4a_macro_f1": results["regenerated_v4a"]["macro_f1"],
                "candidate_i_macro_f1": results["regenerated_candidate_i"][
                    "macro_f1"
                ],
                "candidate_q_macro_f1": results["candidate_q_authenticated"][
                    "macro_f1"
                ],
                "verdict": results["verdict"],
                "raw_text_persisted": False,
                "competition_test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
