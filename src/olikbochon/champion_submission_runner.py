"""Locked CLI for final current-champion inference and submission artifacts."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .champion_submission import (
    EXPECTED_TEST_ROWS,
    fit_frozen_champion,
    frozen_champion_config,
    predict_frozen_champion,
)
from .data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file


MODE = "champion-submission"
OUTPUT_NAME = "current_champion_0692051"
EXPECTED_TEST_FILENAME = "test set.csv"
EXPECTED_TEST_BYTES = 2_329_947
EXPECTED_TEST_SHA256 = "db75049956c6fa00e4d9c476716ee34bc4cc17a737f52ada06d2c0f80d567b81"


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
        raise OSError("Unable to query champion inference memory")
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m olikbochon.champion_submission_runner",
        description="Run locked full-data inference for the current official-only champion.",
    )
    parser.add_argument("--mode", required=True, choices=(MODE,))
    parser.add_argument("--train-path", required=True, type=Path)
    parser.add_argument("--test-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_arguments(
    args: argparse.Namespace, root: Path
) -> tuple[Path, Path, Path]:
    root = Path(root).resolve()
    expected_train = (root / "data" / "competition" / "dataset samples.json").resolve()
    expected_test = (root / "data" / "competition" / EXPECTED_TEST_FILENAME).resolve()
    expected_output = (root / "artifacts" / "submissions" / OUTPUT_NAME).resolve()
    supplied = (args.train_path, args.test_path, args.output_dir)
    if any(not Path(value).is_absolute() for value in supplied):
        raise ValueError("Champion train, test, and output paths must be absolute")
    resolved = tuple(Path(value).resolve() for value in supplied)
    if resolved[0] != expected_train:
        raise ValueError("Champion training path must be the official labeled sample")
    if resolved[1] != expected_test:
        raise ValueError("Champion test path must be the official competition test")
    if resolved[2] != expected_output:
        raise ValueError(f"Champion output must be artifacts/submissions/{OUTPUT_NAME}")
    ignore_lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    if "artifacts/submissions/" not in ignore_lines:
        raise ValueError("artifacts/submissions/ must be explicitly ignored")
    if expected_output.exists():
        raise ValueError("Champion output already exists; overwrite is prohibited")
    return expected_train, expected_test, expected_output


def authenticate_test_file(path: Path) -> dict[str, Any]:
    if path.name != EXPECTED_TEST_FILENAME or not path.is_file():
        raise ValueError("Competition test path is missing or has the wrong filename")
    size = path.stat().st_size
    if size != EXPECTED_TEST_BYTES:
        raise ValueError("Competition test byte size does not match the authenticated file")
    digest = sha256_file(path)
    if digest != EXPECTED_TEST_SHA256:
        raise ValueError("Competition test SHA-256 does not match the authenticated file")
    return {"filename": path.name, "bytes": size, "sha256": digest}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    started = perf_counter()
    args = build_cli_parser().parse_args(argv)
    root = repository_root()
    train_path, test_path, output = validate_arguments(args, root)
    training = load_labeled_json(train_path, expected_sha256=OFFICIAL_SAMPLE_SHA256)
    fitted = fit_frozen_champion(training)
    test_authentication = authenticate_test_file(test_path)
    test = pd.read_csv(test_path)
    if len(test) != EXPECTED_TEST_ROWS:
        raise ValueError("Authenticated competition test must contain exactly 2,516 rows")
    outputs = predict_frozen_champion(fitted, test)
    output.mkdir(parents=True, exist_ok=False)
    submission_path = output / "submission.csv"
    probabilities_path = output / "champion_test_probabilities.csv"
    summary_path = output / "run_summary.json"
    checksums_path = output / "artifact_checksums.json"
    outputs.submission.to_csv(submission_path, index=False)
    outputs.probabilities.to_csv(probabilities_path, index=False)
    primary_hashes = {
        submission_path.name: sha256_file(submission_path),
        probabilities_path.name: sha256_file(probabilities_path),
    }
    memory = process_memory_bytes()
    summary = {
        "git_commit": _git_commit(root),
        "frozen_configuration": frozen_champion_config(),
        "training": fitted.training_audit,
        "test_authentication": test_authentication,
        "inference": outputs.inference_audit,
        "runtime_seconds": perf_counter() - started,
        "process_memory": memory,
        "file_hashes": primary_hashes,
        "competition_rows_printed_or_manually_reviewed": 0,
        "raw_test_text_persisted": False,
    }
    _write_json(summary_path, summary)
    artifact_hashes = {
        **primary_hashes,
        summary_path.name: sha256_file(summary_path),
    }
    _write_json(checksums_path, artifact_hashes)
    print(
        json.dumps(
            {
                "status": "complete",
                "test_rows": outputs.inference_audit["test_rows"],
                "test_route_counts": outputs.inference_audit["test_route_counts"],
                "predicted_label_counts": outputs.inference_audit[
                    "predicted_label_counts"
                ],
                "raw_test_text_printed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
