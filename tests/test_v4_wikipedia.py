from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olikbochon import v4_wikipedia as v4
from olikbochon.data_loading import DataValidationError


def labeled_frame(group_count: int = 15) -> pd.DataFrame:
    records = []
    for group in range(group_count):
        for label in (0, 1):
            records.append(
                {
                    "context": "" if group % 2 == 0 else f"প্রসঙ্গ {group} তথ্য",
                    "prompt_bn": f"প্রশ্ন {group}",
                    "response_bn": f"উত্তর {group} তথ্য {label}",
                    "label": label,
                }
            )
    return pd.DataFrame(records)


def test_lexical_features_reproduce_number_and_overlap_semantics() -> None:
    features = v4.make_lexical_features("ঢাকা ২০২৪ সালে বড় শহর", "ঢাকা 2024")
    assert tuple(features) == v4.FEATURE_COLUMNS
    assert features["ctx_present"] == 1
    assert features["word_overlap_ratio"] == 1.0
    assert features["numbers_supported"] == 1
    assert features["number_overlap_ratio"] == 1.0
    assert features["has_numbers_in_response"] == 1

    missing = v4.make_lexical_features("", "সংখ্যা 7")
    assert missing["ctx_present"] == 0
    assert missing["numbers_supported"] == 1
    assert missing["number_overlap_ratio"] == 0.0


def test_duplicate_groups_keep_prompt_context_family_together() -> None:
    frame = labeled_frame(8)
    audit = v4.build_duplicate_groups(frame)
    assert audit.group_count == 8
    assert audit.nontrivial_groups == 8
    folds = v4.make_grouped_folds(
        frame["label"].to_numpy(), audit.group_ids, n_splits=4, seed=42
    )
    groups = np.asarray(audit.group_ids)
    for train, validation in folds.folds:
        assert not (set(groups[train]) & set(groups[validation]))


def test_exact_duplicate_label_conflict_is_rejected() -> None:
    frame = labeled_frame(5)
    duplicate = frame.iloc[[0]].copy()
    duplicate["label"] = 1
    with pytest.raises(DataValidationError, match="Conflicting labels"):
        v4.build_duplicate_groups(pd.concat([frame, duplicate], ignore_index=True))


def test_original_and_honest_grouped_estimates_are_separate() -> None:
    frame = labeled_frame()
    retrieval = v4.RetrievalResult(
        tuple(f"উইকিপিডিয়া প্রসঙ্গ {index} তথ্য 0 তথ্য 1" for index in range(len(frame))),
        np.linspace(0.1, 0.4, len(frame)),
    )
    results = v4.evaluate_v4a(frame, retrieval)
    assert results.original_method["optimistic"] is True
    assert "full-OOF" in results.original_method["name"]
    assert results.grouped_method["optimistic"] is False
    assert "nested 5x3" in results.grouped_method["name"]
    assert len(results.grouped_oof_probabilities) == len(frame)
    assert np.isfinite(results.grouped_oof_probabilities).all()
    assert set(results.grouped_fold_by_row) == set(range(5))


def test_probability_artifacts_require_ordered_safe_ids() -> None:
    oof = pd.DataFrame(
        {
            "row_index": [0, 1],
            "fold": [0, 1],
            "label": [0, 1],
            "probability_label1": [0.2, 0.8],
        }
    )
    v4.validate_oof_probability_artifact(oof, 2)
    test_ids = pd.Series(["b", "a"])
    artifact = pd.DataFrame({"id": ["b", "a"], "probability_label1": [0.1, 0.9]})
    v4.validate_test_probability_artifact(artifact, test_ids)
    with pytest.raises(DataValidationError, match="preserve test order"):
        v4.validate_test_probability_artifact(artifact.iloc[::-1].reset_index(drop=True), test_ids)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _manifest_digest(root: Path, paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{v4.sha256_file(path)}  ./{relative}\n".encode())
    return digest.hexdigest()


def synthetic_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    competition = tmp_path / "competitions" / "challenge"
    official = _write(competition / v4.OFFICIAL_FILENAME, b"official")
    test = _write(competition / v4.TEST_FILENAME, b"test")
    sample = _write(competition / "sample submission.csv", b"sample")
    wiki = tmp_path / "datasets" / "abyaadrafid" / "bnwiki"
    chunks = (
        _write(wiki / "lolol" / "AA" / "wiki_00", b"first"),
        _write(wiki / "lolol" / "lolol" / "AA" / "wiki_00", b"first"),
    )
    monkeypatch.setattr(v4, "EXPECTED_WIKI_RELATIVE_NAMES", frozenset({
        "lolol/AA/wiki_00", "lolol/lolol/AA/wiki_00"
    }))
    monkeypatch.setattr(v4, "WIKI_FILE_COUNT", 2)
    monkeypatch.setattr(v4, "WIKI_TOTAL_SIZE", 10)
    monkeypatch.setattr(v4, "WIKI_CONTENT_MANIFEST_SHA256", _manifest_digest(wiki, chunks))
    monkeypatch.setattr(
        v4,
        "KNOWN_FILE_HASHES",
        {
            v4.OFFICIAL_FILENAME: v4.sha256_file(official),
            v4.TEST_FILENAME: v4.sha256_file(test),
            "sample submission.csv": v4.sha256_file(sample),
        },
    )
    return competition, wiki


def test_nested_input_discovery_authenticates_official_and_wikipedia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    competition, wiki = synthetic_inputs(tmp_path, monkeypatch)
    files = v4.discover_v4_files(tmp_path)
    assert files.competition_root == competition
    assert files.wikipedia_root == wiki
    assert len(files.wikipedia_chunks) == 2


def test_competition_ambiguity_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic_inputs(tmp_path, monkeypatch)
    second = tmp_path / "competitions" / "second"
    _write(second / v4.OFFICIAL_FILENAME, b"official")
    _write(second / v4.TEST_FILENAME, b"test")
    _write(second / "sample submission.csv", b"sample")
    with pytest.raises(v4.V4DiscoveryError, match="More than one coherent"):
        v4.discover_v4_files(tmp_path)
