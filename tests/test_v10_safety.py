from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from olikbochon.v10_runner import PRIMARY_REVISION, validate_arguments


def _arguments(root: Path) -> Namespace:
    return Namespace(
        mode="nested-lexical-semantic-ensemble",
        corpus_path=(root / "data" / "retrieval" / "bnwiki").resolve(),
        primary_model_path=(
            root / "data" / "models" / "v9_nli" / f"mdeberta_xnli@{PRIMARY_REVISION}"
        ).resolve(),
        seeds=[17, 29, 43],
        folds=5,
        cpu_workers=4,
        outer_workers=3,
        output_dir=(root / "artifacts" / "v10" / "nested_lexical_semantic_ensemble").resolve(),
    )


def test_v10_cli_accepts_only_frozen_local_paths(tmp_path: Path) -> None:
    args = _arguments(tmp_path)
    args.corpus_path.mkdir(parents=True)
    args.primary_model_path.mkdir(parents=True)
    (tmp_path / ".gitignore").write_text("artifacts/v10/\n", encoding="utf-8")
    corpus, primary, output = validate_arguments(args, tmp_path)
    assert corpus == args.corpus_path
    assert primary == args.primary_model_path
    assert output == args.output_dir
    args.output_dir = (tmp_path / "outside").resolve()
    with pytest.raises(ValueError, match="approved frozen path"):
        validate_arguments(args, tmp_path)


def test_v10_cli_rejects_changed_workers_or_outer_folds() -> None:
    root = Path.cwd()
    args = _arguments(root)
    args.outer_workers = 2
    with pytest.raises(ValueError, match="three outer workers"):
        validate_arguments(args, root)
    args.outer_workers = 3
    args.folds = 4
    with pytest.raises(ValueError, match="five outer folds"):
        validate_arguments(args, root)
