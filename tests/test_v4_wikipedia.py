from __future__ import annotations

import json
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


def test_wikipedia_discovery_policy_is_dynamic_with_a_safety_floor() -> None:
    assert v4.MIN_WIKI_CHUNK_COUNT == 250
    assert v4.WIKI_SECTION_NAMES == ("AA", "AB", "AC", "AD")
    assert v4.WIKI_CHUNK_PATTERN.fullmatch("wiki_00")
    assert not v4.WIKI_CHUNK_PATTERN.fullmatch("wiki_00.json")


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


def _wiki_record(name: str) -> bytes:
    payload = {
        "url": f"https://bn.wikipedia.org/wiki/{name}",
        "text": f"শিরোনাম {name}\n\n" + "বাংলা বিশ্বকোষের যাচাইযোগ্য নিবন্ধের লেখা। " * 3,
    }
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode()


def synthetic_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    competition = tmp_path / "competitions" / "challenge"
    official = _write(competition / v4.OFFICIAL_FILENAME, b"official")
    test = _write(competition / v4.TEST_FILENAME, b"test")
    sample = _write(competition / "sample submission.csv", b"sample")
    wiki = tmp_path / "datasets" / "abyaadrafid" / "bnwiki" / "lolol"
    _write(wiki / "AA" / "wiki_00", _wiki_record("first"))
    _write(wiki / "AD" / "wiki_00", _wiki_record("other"))
    (wiki / "AB").mkdir()
    (wiki / "AC").mkdir()
    monkeypatch.setattr(v4, "MIN_WIKI_CHUNK_COUNT", 2)
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
    assert files.wikipedia_duplicate_path_count == 0
    summary = v4.safe_discovery_summary(files)
    assert summary["wikipedia_chunk_count"] == 2
    assert summary["wikipedia_total_size"] == sum(
        path.stat().st_size for path in files.wikipedia_chunks
    )


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


def test_official_competition_hash_validation_remains_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    competition, _ = synthetic_inputs(tmp_path, monkeypatch)
    _write(competition / v4.TEST_FILENAME, b"changed-after-pin")
    with pytest.raises(v4.V4DiscoveryError, match="Official file digest mismatch"):
        v4.discover_v4_files(tmp_path)


def test_byte_identical_duplicate_archive_tree_is_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, wiki = synthetic_inputs(tmp_path, monkeypatch)
    duplicate = wiki / "lolol"
    _write(duplicate / "AA" / "wiki_00", _wiki_record("first"))
    _write(duplicate / "AD" / "wiki_00", _wiki_record("other"))
    (duplicate / "AB").mkdir()
    (duplicate / "AC").mkdir()

    files = v4.discover_v4_files(tmp_path)

    assert files.wikipedia_root == wiki
    assert len(files.wikipedia_chunks) == 2
    assert files.wikipedia_duplicate_path_count == 2


def test_distinct_duplicate_archive_tree_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, wiki = synthetic_inputs(tmp_path, monkeypatch)
    duplicate = wiki / "lolol"
    _write(duplicate / "AA" / "wiki_00", _wiki_record("changed"))
    _write(duplicate / "AD" / "wiki_00", _wiki_record("other"))
    (duplicate / "AB").mkdir()
    (duplicate / "AC").mkdir()
    with pytest.raises(v4.V4DiscoveryError, match="More than one distinct"):
        v4.discover_v4_files(tmp_path)


def test_wikipedia_discovery_rejects_too_few_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synthetic_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(v4, "MIN_WIKI_CHUNK_COUNT", 3)
    with pytest.raises(v4.V4DiscoveryError, match="at least 3"):
        v4.discover_v4_files(tmp_path)


def test_wikipedia_chunks_parse_strictly_before_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _ = synthetic_inputs(tmp_path, monkeypatch)
    files = v4.discover_v4_files(tmp_path)

    corpus = v4.load_wikipedia_corpus(files)

    assert corpus.source_chunk_count == 2
    assert corpus.decoded_line_count == 2
    assert corpus.rejected_line_count == 0
    assert len(corpus.articles) == 2


@pytest.mark.parametrize(
    "invalid_payload, message",
    [
        (b"not json\n", "invalid JSON"),
        (b"[]\n", "non-object"),
        (b'{"url":"x","text":7}\n', "non-string text"),
        (b"\n", "contains no JSON records"),
    ],
)
def test_wikipedia_parse_validation_rejects_invalid_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_payload: bytes,
    message: str,
) -> None:
    _, wiki = synthetic_inputs(tmp_path, monkeypatch)
    _write(wiki / "AA" / "wiki_00", invalid_payload)
    files = v4.discover_v4_files(tmp_path)

    with pytest.raises(DataValidationError, match=message):
        v4.load_wikipedia_corpus(files)
