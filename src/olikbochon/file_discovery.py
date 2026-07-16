"""Deterministic Kaggle input discovery using filenames and hashes only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from olikbochon.data_loading import sha256_file


TRAIN_NAMES = ("dataset samples.json",)
TEST_NAMES = ("test set.csv",)
SAMPLE_SUBMISSION_NAMES = ("sample submission.csv", "sample_submission.csv")


class DiscoveryError(RuntimeError):
    """Raised when mandatory inputs cannot be selected safely."""


@dataclass(frozen=True)
class DiscoveredFiles:
    train: Path
    test: Path
    sample_submission: Path | None


def discover_preferred_file(
    root: Path, accepted_names: tuple[str, ...], *, mandatory: bool
) -> Path | None:
    """Find accepted names recursively and reject non-identical ambiguity."""
    root = Path(root)
    candidates: list[tuple[int, Path]] = []
    for preference, name in enumerate(accepted_names):
        candidates.extend((preference, path) for path in sorted(root.rglob(name)) if path.is_file())
    if not candidates:
        if mandatory:
            raise DiscoveryError(
                f"Missing mandatory file under {root}: accepted names={list(accepted_names)}"
            )
        return None

    hashes = {sha256_file(path) for _, path in candidates}
    if len(hashes) != 1:
        metadata = [f"{path} ({sha256_file(path)})" for _, path in candidates]
        raise DiscoveryError("Non-identical accepted candidates found: " + "; ".join(metadata))
    return min(candidates, key=lambda item: (item[0], str(item[1])))[1]


def discover_kaggle_files(root: Path = Path("/kaggle/input")) -> DiscoveredFiles:
    """Discover mandatory training/test files and an optional structural template."""
    return DiscoveredFiles(
        train=discover_preferred_file(root, TRAIN_NAMES, mandatory=True),
        test=discover_preferred_file(root, TEST_NAMES, mandatory=True),
        sample_submission=discover_preferred_file(
            root, SAMPLE_SUBMISSION_NAMES, mandatory=False
        ),
    )
