from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olikbochon.data_loading import DataValidationError
from olikbochon.v3_data import (
    KNOWN_HASHES,
    MODEL_REQUIRED_FILES,
    audit_partition_safety,
    authenticate_known_file,
    authenticate_model_directory,
    build_official_groups,
    discover_v3_files,
    load_labeled_json,
    make_official_folds,
)


def labeled(rows: int = 10) -> pd.DataFrame:
    values = []
    for index in range(rows):
        values.append(
            {
                "context": "তথ্য" if index % 3 else None,
                "prompt_bn": f"প্রশ্ন {index}",
                "response_bn": f"উত্তর {index}",
                "label": index % 2,
            }
        )
    return pd.DataFrame(values, columns=["context", "prompt_bn", "response_bn", "label"])


def write_json(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_json(orient="records", force_ascii=False), encoding="utf-8")


def write_model(path: Path, weight: bytes) -> None:
    path.mkdir(parents=True)
    for name in MODEL_REQUIRED_FILES:
        if name == "pytorch_model.bin":
            (path / name).write_bytes(weight)
        elif name == "config.json":
            (path / name).write_text(
                json.dumps({"model_type": "electra", "vocab_size": 32000}), encoding="utf-8"
            )
        else:
            (path / name).write_text(name, encoding="utf-8")


def test_dynamic_labeled_row_counts_and_known_hash_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dynamic.json"
    write_json(path, labeled(7))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setitem(KNOWN_HASHES, path.name, digest)
    assert len(load_labeled_json(path)) == 7
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DataValidationError, match="digest mismatch"):
        authenticate_known_file(path)


def test_exact_source_schema_is_required(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    frame = labeled(4).rename(columns={"prompt_bn": "prompt"})
    write_json(path, frame)
    with pytest.raises(DataValidationError, match="Expected columns"):
        load_labeled_json(path)


def test_model_authentication_requires_exact_digest_and_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import olikbochon.v3_data as module

    weight = b"synthetic-weight"
    model = tmp_path / "model"
    write_model(model, weight)
    monkeypatch.setattr(module, "MODEL_WEIGHT_SHA256", hashlib.sha256(weight).hexdigest())
    assert authenticate_model_directory(model)["required_file_count"] == 7
    (model / "extra.py").write_text("pass", encoding="utf-8")
    with pytest.raises(Exception, match="unexpected entries"):
        authenticate_model_directory(model)


def test_discovery_quarantines_public_csv_and_selects_coherent_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import olikbochon.v3_data as module

    public = tmp_path / "public"
    competition = tmp_path / "competition"
    write_json(public / "bangla_hallucination_5k_train.json", labeled(6))
    write_json(public / "bangla_hallucination_5k_validation.json", labeled(4))
    (public / "test set.csv").write_text("id,context,prompt_bn,response_bn\n", encoding="utf-8")
    write_json(competition / "dataset samples.json", labeled(10))
    competition.mkdir(exist_ok=True)
    (competition / "test set.csv").write_text(
        "id,context,prompt_bn,response_bn\n1,,,\n", encoding="utf-8"
    )
    (competition / "sample submission.csv").write_text("id,label\n1,0\n", encoding="utf-8")
    model = tmp_path / "model"
    weight = b"weight"
    write_model(model, weight)
    monkeypatch.setattr(module, "MODEL_WEIGHT_SHA256", hashlib.sha256(weight).hexdigest())
    files = discover_v3_files(tmp_path)
    assert files.test.parent == competition
    assert files.test.parent != public
    (tmp_path / "unexpected-fourth-input").mkdir()
    with pytest.raises(Exception, match="exactly three"):
        discover_v3_files(tmp_path)


def test_partition_overlap_and_conflict_are_fatal() -> None:
    train = labeled(10)
    validation = labeled(6).assign(prompt_bn=lambda frame: frame["prompt_bn"] + " v")
    official = labeled(8).assign(prompt_bn=lambda frame: frame["prompt_bn"] + " o")
    audit_partition_safety(
        {"public_train": train, "public_validation": validation, "official": official}
    )
    official.loc[0, ["prompt_bn", "context", "response_bn"]] = train.loc[
        0, ["prompt_bn", "context", "response_bn"]
    ]
    official.loc[0, "label"] = 1 - int(train.loc[0, "label"])
    with pytest.raises(DataValidationError, match="conflicting labels"):
        audit_partition_safety(
            {"public_train": train, "public_validation": validation, "official": official}
        )


def test_group_construction_and_folds_are_deterministic_without_leakage() -> None:
    frame = labeled(30)
    frame.loc[1, ["prompt_bn", "context"]] = frame.loc[0, ["prompt_bn", "context"]]
    first = build_official_groups(frame)
    second = build_official_groups(frame)
    assert first == second
    assert first.nontrivial_groups >= 1
    folds = make_official_folds(frame["label"].to_numpy(dtype=np.int64), first)
    repeated = make_official_folds(frame["label"].to_numpy(dtype=np.int64), first)
    assert folds.validation_fold_by_row == repeated.validation_fold_by_row
    assert folds.strategy == "stratified_group_5fold"
    groups = np.asarray(first.group_ids)
    for train, validation in folds.folds:
        assert not (set(groups[train]) & set(groups[validation]))


def test_plain_stratification_requires_no_nontrivial_groups() -> None:
    frame = labeled(30)
    audit = build_official_groups(frame)
    assert audit.nontrivial_groups == 0
    folds = make_official_folds(frame["label"].to_numpy(dtype=np.int64), audit)
    assert folds.strategy == "stratified_5fold_after_no_group_audit"
