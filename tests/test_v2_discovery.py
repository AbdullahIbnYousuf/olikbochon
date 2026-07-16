from __future__ import annotations

from pathlib import Path

import pytest

from olikbochon.file_discovery import DiscoveryError
from olikbochon.v2_data import (
    OFFICIAL_FILENAME,
    PUBLIC_AGGREGATE_FILENAME,
    PUBLIC_TRAIN_FILENAME,
    PUBLIC_VALIDATION_FILENAME,
)
from olikbochon.v2_discovery import TEST_FILENAME, discover_v2_files


def write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def fixture_tree(root: Path) -> tuple[Path, Path]:
    public = root / "arbitrary-public-name"
    competition = root / "arbitrary-competition-name"
    write(public / PUBLIC_TRAIN_FILENAME, b"train")
    write(public / PUBLIC_VALIDATION_FILENAME, b"validation")
    write(public / PUBLIC_AGGREGATE_FILENAME, b"aggregate")
    write(public / TEST_FILENAME, b"identical-test")
    write(competition / OFFICIAL_FILENAME, b"official")
    write(competition / "sample submission.csv", b"sample")
    write(competition / TEST_FILENAME, b"identical-test")
    return public, competition


def test_public_test_copy_is_quarantined_and_competition_test_selected(tmp_path: Path) -> None:
    public, competition = fixture_tree(tmp_path)
    files = discover_v2_files(tmp_path)
    assert files.test == competition / TEST_FILENAME
    assert files.test != public / TEST_FILENAME
    assert files.competition_root == competition
    assert public in files.public_roots


def test_filename_and_identical_hash_cannot_make_public_csv_selectable(tmp_path: Path) -> None:
    public, competition = fixture_tree(tmp_path)
    assert (public / TEST_FILENAME).read_bytes() == (competition / TEST_FILENAME).read_bytes()
    assert discover_v2_files(tmp_path).test.parent == competition


def test_aggregate_is_not_selected_as_a_training_role(tmp_path: Path) -> None:
    public, _ = fixture_tree(tmp_path)
    files = discover_v2_files(tmp_path)
    assert files.public_train == public / PUBLIC_TRAIN_FILENAME
    assert files.public_validation == public / PUBLIC_VALIDATION_FILENAME
    assert files.public_aggregate == public / PUBLIC_AGGREGATE_FILENAME


def test_preferred_sample_submission_name_wins_within_one_root(tmp_path: Path) -> None:
    _, competition = fixture_tree(tmp_path)
    write(competition / "sample_submission.csv", b"different-secondary-template")
    files = discover_v2_files(tmp_path)
    assert files.sample_submission == competition / "sample submission.csv"


def test_conflicting_competition_roots_fail(tmp_path: Path) -> None:
    fixture_tree(tmp_path)
    second = tmp_path / "second-competition"
    write(second / OFFICIAL_FILENAME, b"different-official")
    write(second / "sample submission.csv", b"sample")
    write(second / TEST_FILENAME, b"different-test")
    with pytest.raises(DiscoveryError, match="Conflicting non-identical coherent"):
        discover_v2_files(tmp_path)


def test_missing_public_split_fails(tmp_path: Path) -> None:
    write(tmp_path / "public" / PUBLIC_TRAIN_FILENAME, b"train")
    with pytest.raises(DiscoveryError, match="both approved public 4k and 1k"):
        discover_v2_files(tmp_path)
