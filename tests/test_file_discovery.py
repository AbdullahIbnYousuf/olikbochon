from pathlib import Path

import pytest

from olikbochon.file_discovery import DiscoveryError, discover_preferred_file


def write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_identical_duplicates_are_selected_deterministically(tmp_path: Path) -> None:
    expected = write_bytes(tmp_path / "a" / "dataset samples.json", b"same")
    write_bytes(tmp_path / "b" / "dataset samples.json", b"same")
    selected = discover_preferred_file(tmp_path, ("dataset samples.json",), mandatory=True)
    assert selected == expected


def test_nonidentical_duplicates_are_rejected(tmp_path: Path) -> None:
    write_bytes(tmp_path / "a" / "test set.csv", b"first")
    write_bytes(tmp_path / "b" / "test set.csv", b"second")
    with pytest.raises(DiscoveryError, match="Non-identical"):
        discover_preferred_file(tmp_path, ("test set.csv",), mandatory=True)


def test_organizer_sample_submission_name_has_preference(tmp_path: Path) -> None:
    preferred = write_bytes(tmp_path / "z" / "sample submission.csv", b"same")
    write_bytes(tmp_path / "a" / "sample_submission.csv", b"same")
    selected = discover_preferred_file(
        tmp_path,
        ("sample submission.csv", "sample_submission.csv"),
        mandatory=False,
    )
    assert selected == preferred


def test_missing_mandatory_and_optional_files(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError, match="Missing mandatory"):
        discover_preferred_file(tmp_path, ("missing.csv",), mandatory=True)
    assert discover_preferred_file(tmp_path, ("missing.csv",), mandatory=False) is None
