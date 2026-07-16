"""Role-aware Version 2 Kaggle discovery that quarantines public CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from olikbochon.data_loading import sha256_file
from olikbochon.file_discovery import DiscoveryError
from olikbochon.v2_data import (
    OFFICIAL_FILENAME,
    PUBLIC_AGGREGATE_FILENAME,
    PUBLIC_TRAIN_FILENAME,
    PUBLIC_VALIDATION_FILENAME,
)


TEST_FILENAME = "test set.csv"
SAMPLE_SUBMISSION_NAMES = ("sample submission.csv", "sample_submission.csv")


@dataclass(frozen=True)
class V2DiscoveredFiles:
    """Resolved public labeled files and coherent competition-root files."""

    public_train: Path
    public_validation: Path
    public_aggregate: Path
    official_train: Path
    test: Path
    sample_submission: Path
    public_roots: tuple[Path, ...]
    competition_root: Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _identical_paths(paths: list[Path]) -> bool:
    return len({sha256_file(path) for path in paths}) == 1


def _public_roots(root: Path) -> list[Path]:
    train_parents = {path.parent for path in root.rglob(PUBLIC_TRAIN_FILENAME) if path.is_file()}
    validation_parents = {
        path.parent for path in root.rglob(PUBLIC_VALIDATION_FILENAME) if path.is_file()
    }
    roots = sorted(train_parents.intersection(validation_parents), key=str)
    if not roots:
        raise DiscoveryError("No directory contains both approved public 4k and 1k files")
    return roots


def _select_public_copy(roots: list[Path], filename: str) -> Path:
    paths = [root / filename for root in roots]
    if not all(path.is_file() for path in paths):
        raise DiscoveryError(f"Public roots do not all contain {filename!r}")
    if not _identical_paths(paths):
        raise DiscoveryError(f"Conflicting non-identical public copies of {filename!r}")
    return min(paths, key=str)


def _competition_roots(root: Path, public_roots: list[Path]) -> list[tuple[int, Path]]:
    candidate_roots: set[Path] = set()
    for sample_name in SAMPLE_SUBMISSION_NAMES:
        for sample in sorted(root.rglob(sample_name)):
            candidate = sample.parent
            if any(_is_within(candidate, public_root) for public_root in public_roots):
                continue
            if (candidate / OFFICIAL_FILENAME).is_file() and (
                candidate / TEST_FILENAME
            ).is_file():
                candidate_roots.add(candidate)
    return [
        (
            next(
                preference
                for preference, sample_name in enumerate(SAMPLE_SUBMISSION_NAMES)
                if (candidate / sample_name).is_file()
            ),
            candidate,
        )
        for candidate in sorted(candidate_roots, key=str)
    ]


def _competition_signature(candidate: tuple[int, Path]) -> tuple[str, str, str]:
    preference, root = candidate
    sample = root / SAMPLE_SUBMISSION_NAMES[preference]
    return (
        sha256_file(root / OFFICIAL_FILENAME),
        sha256_file(root / TEST_FILENAME),
        sha256_file(sample),
    )


def discover_v2_files(root: Path = Path("/kaggle/input")) -> V2DiscoveredFiles:
    """Resolve V2 inputs while excluding every CSV below public-data roots."""
    root = Path(root)
    public_roots = _public_roots(root)
    public_train = _select_public_copy(public_roots, PUBLIC_TRAIN_FILENAME)
    public_validation = _select_public_copy(public_roots, PUBLIC_VALIDATION_FILENAME)
    aggregate_paths = [path / PUBLIC_AGGREGATE_FILENAME for path in public_roots]
    existing_aggregates = [path for path in aggregate_paths if path.is_file()]
    if len(existing_aggregates) != len(public_roots) or not _identical_paths(
        existing_aggregates
    ):
        raise DiscoveryError("Missing, conflicting, or incomplete public aggregate copies")
    public_aggregate = min(existing_aggregates, key=str)

    candidates = _competition_roots(root, public_roots)
    if not candidates:
        raise DiscoveryError(
            "No coherent non-public competition root contains official sample, test, and template"
        )
    signatures = {_competition_signature(candidate) for candidate in candidates}
    if len(signatures) != 1:
        raise DiscoveryError("Conflicting non-identical coherent competition roots")
    preference, competition_root = min(candidates, key=lambda item: (item[0], str(item[1])))
    sample_submission = competition_root / SAMPLE_SUBMISSION_NAMES[preference]
    test = competition_root / TEST_FILENAME
    if any(_is_within(test, public_root) for public_root in public_roots):
        raise DiscoveryError("Resolved test file is inside a quarantined public-data directory")

    return V2DiscoveredFiles(
        public_train=public_train,
        public_validation=public_validation,
        public_aggregate=public_aggregate,
        official_train=competition_root / OFFICIAL_FILENAME,
        test=test,
        sample_submission=sample_submission,
        public_roots=tuple(public_roots),
        competition_root=competition_root,
    )
