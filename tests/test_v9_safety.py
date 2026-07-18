from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from olikbochon.v9_runner import (
    CONTROL_REVISION,
    PRIMARY_REVISION,
    memory_status,
    validate_arguments,
)


def _arguments(root: Path) -> Namespace:
    return Namespace(
        mode="frozen-nli-verification",
        corpus_path=(root / "data" / "retrieval" / "bnwiki").resolve(),
        primary_model_path=(
            root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{PRIMARY_REVISION}"
        ).resolve(),
        control_model_path=(
            root / "data" / "models" / "v9_nli" / f"xlmr_large_xnli@{CONTROL_REVISION}"
        ).resolve(),
        seeds=[17, 29, 43],
        folds=5,
        cpu_workers=4,
        fold_workers=3,
        output_dir=(root / "artifacts" / "v9" / "frozen_nli_verification").resolve(),
    )


def test_cli_paths_are_restricted_to_approved_local_locations(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    args.corpus_path.mkdir(parents=True)
    args.primary_model_path.mkdir(parents=True)
    args.control_model_path.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text("artifacts/v9/\n", encoding="utf-8")
    corpus, primary, control, output = validate_arguments(args, tmp_path)
    assert corpus == args.corpus_path
    assert primary == args.primary_model_path
    assert control == args.control_model_path
    assert output == args.output_dir
    args.output_dir = (tmp_path / "outside").resolve()
    with pytest.raises(ValueError, match="approved frozen path"):
        validate_arguments(args, tmp_path)


def test_cli_rejects_changed_workers_or_validation() -> None:
    root = Path.cwd()
    args = _arguments(root)
    args.cpu_workers = 3
    with pytest.raises(ValueError, match="four preprocessing"):
        validate_arguments(args, root)
    args.cpu_workers = 4
    args.seeds = [17]
    with pytest.raises(ValueError, match="seeds 17 29 43"):
        validate_arguments(args, root)


def test_windows_memory_audit_returns_bounded_numeric_values() -> None:
    record = memory_status()
    assert 0.0 <= record["system_memory_fraction"] <= 1.0
    assert record["system_total_physical_bytes"] > 0
    assert record["process_working_set_bytes"] > 0
