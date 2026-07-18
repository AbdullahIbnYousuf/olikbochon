from __future__ import annotations

import inspect

from olikbochon import v12_runner
from olikbochon.v12_qwen_judge import parse_compact_json


def test_compact_json_parser_repairs_without_model_retry() -> None:
    assert parse_compact_json('{"label":1,"confidence":0.75}') == (1, 0.75)
    assert parse_compact_json("{'label':0,'confidence':0.6}") == (0, 0.6)
    assert parse_compact_json("not json") is None


def test_runner_freezes_winner_before_holdout_targets() -> None:
    source = inspect.getsource(v12_runner.main)
    assert source.index("frozen_winner = winner") < source.index("holdout_targets =")
    assert source.index("holdout_targets =") < source.index("test_path =")
    assert '"holdout_used_for_selection": False' in source
    assert '"holdout_manual_review": False' in source


def test_public_contrastive_is_authenticated_but_not_pooled() -> None:
    source = inspect.getsource(v12_runner._load_public)
    assert "CONTRASTIVE_AUTH" in source
    assert "excluded_contrastive" in source
