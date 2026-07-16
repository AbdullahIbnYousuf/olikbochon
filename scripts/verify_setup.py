#!/usr/bin/env python3
"""Verify setup artifacts without reading or displaying raw competition rows."""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Artifact:
    name: str
    directory: Path
    exact_names: tuple[str, ...]
    fallback_globs: tuple[str, ...]
    mandatory: bool = True
    kind: str = "file"


ARTIFACTS = (
    Artifact(
        "original starter notebook",
        ROOT / "notebooks" / "original",
        ("starter-notebook-datathon.ipynb",),
        ("*.ipynb",),
        kind="notebook",
    ),
    Artifact(
        "public 20k training JSON",
        ROOT / "data" / "public-20k",
        ("hallucination_detection_bn_20k.json",),
        ("*20k*.json", "*.json"),
    ),
    Artifact(
        "competition sample JSON",
        ROOT / "data" / "competition",
        ("dataset samples.json", "dataset_samples.json"),
        ("*dataset*sample*.json", "*sample*.json"),
    ),
    Artifact(
        "competition test CSV",
        ROOT / "data" / "competition",
        ("test set.csv", "test_set.csv", "test.csv"),
        ("*test*.csv",),
        kind="csv",
    ),
    Artifact(
        "sample submission CSV",
        ROOT / "data" / "competition",
        ("sample_submission.csv",),
        ("*sample*submission*.csv",),
        mandatory=False,
        kind="csv",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate(artifact: Artifact) -> Path | None:
    for name in artifact.exact_names:
        candidate = artifact.directory / name
        if candidate.is_file():
            return candidate

    seen: set[Path] = set()
    for pattern in artifact.fallback_globs:
        for candidate in sorted(artifact.directory.rglob(pattern)):
            if candidate.is_file() and candidate not in seen:
                seen.add(candidate)
                return candidate
    return None


def validate_notebook(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        nbformat.read(handle, as_version=4)


def csv_headers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError("CSV has no header") from exc


def main() -> int:
    failures: list[str] = []
    print(f"Project: {ROOT}")

    for artifact in ARTIFACTS:
        requirement = "MANDATORY" if artifact.mandatory else "OPTIONAL"
        path = locate(artifact)
        if path is None:
            print(f"[{requirement}] MISSING: {artifact.name}")
            if artifact.mandatory:
                failures.append(artifact.name)
            continue

        try:
            if artifact.kind == "notebook":
                validate_notebook(path)
            headers = csv_headers(path) if artifact.kind == "csv" else None
            relative = path.relative_to(ROOT)
            print(
                f"[{requirement}] OK: {artifact.name}: {relative} "
                f"({path.stat().st_size} bytes, sha256={sha256(path)})"
            )
            if headers is not None:
                print(f"  CSV headers only: {headers}")
        except (OSError, UnicodeError, ValueError, nbformat.reader.NotJSONError) as exc:
            print(f"[{requirement}] INVALID: {artifact.name}: {type(exc).__name__}: {exc}")
            if artifact.mandatory:
                failures.append(artifact.name)

    if failures:
        print("Setup verification FAILED. Missing or invalid mandatory files:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Setup verification PASSED. All mandatory setup files are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

