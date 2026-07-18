from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from olikbochon.v9_nli import _aggregate_features, authenticate_snapshot, label_indices


def test_label_orientation_is_authenticated_from_both_config_mappings() -> None:
    config = SimpleNamespace(
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
    )
    assert label_indices(config) == {"entailment": 0, "neutral": 1, "contradiction": 2}


def test_label_orientation_rejects_ambiguous_or_conflicting_config() -> None:
    ambiguous = SimpleNamespace(label2id={"LABEL_0": 0}, id2label={0: "LABEL_0"})
    with pytest.raises(ValueError, match="ambiguous"):
        label_indices(ambiguous)
    conflicting = SimpleNamespace(
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
        id2label={0: "contradiction", 1: "neutral", 2: "entailment"},
    )
    with pytest.raises(ValueError, match="conflict"):
        label_indices(conflicting)


def test_frozen_nli_aggregates_use_all_five_passages() -> None:
    probabilities = np.asarray(
        [
            [
                [0.8, 0.1, 0.1],
                [0.7, 0.1, 0.2],
                [0.6, 0.1, 0.3],
                [0.4, 0.1, 0.5],
                [0.2, 0.1, 0.7],
            ]
        ],
        dtype=np.float64,
    )
    scores = np.asarray([[5.0, 4.0, 3.0, 2.0, 1.0]], dtype=np.float64)
    matrix = _aggregate_features(
        probabilities,
        scores,
        {"entailment": 0, "neutral": 1, "contradiction": 2},
    )
    assert matrix.shape == (1, 7)
    assert matrix[0, 0] == pytest.approx(0.8)
    assert matrix[0, 1] == pytest.approx(0.7)
    assert matrix[0, 2] == pytest.approx(0.7)
    assert matrix[0, 3] == pytest.approx(0.5)
    assert matrix[0, 4] == pytest.approx(0.7)
    assert matrix[0, 5] == pytest.approx(9.6 / 15.0)
    assert matrix[0, 6] == 3.0


def test_snapshot_authentication_checks_every_file(tmp_path: Path) -> None:
    files = {"config.json": b"{}\n", "model.safetensors": b"safe-weights"}
    entries = []
    canonical = []
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        entries.append({"path": name, "size": len(raw), "sha256": digest})
        canonical.append(f"{name}\0{len(raw)}\0{digest}\n")
    manifest = {
        "repository_id": "owner/model",
        "revision": "a" * 40,
        "license": "mit",
        "files": entries,
        "content_manifest_sha256": hashlib.sha256("".join(canonical).encode()).hexdigest(),
    }
    (tmp_path / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity = authenticate_snapshot(tmp_path)
    assert identity.revision == "a" * 40
    assert identity.weight_sha256 == hashlib.sha256(files["model.safetensors"]).hexdigest()
    (tmp_path / "config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        authenticate_snapshot(tmp_path)
