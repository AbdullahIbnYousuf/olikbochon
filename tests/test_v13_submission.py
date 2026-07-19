from __future__ import annotations

import inspect

from olikbochon import v13_runner


def test_submission_generation_is_strictly_gated_and_bounded() -> None:
    source = inspect.getsource(v13_runner.main)
    assert source.index("if gate_a or gate_b:") < source.index('"submission_v13_transfer_fallback.csv"')
    assert source.index("if xc_gate:") < source.index('"submission_v13_public_banglabert.csv"')
    assert source.count("build_submission(") == 2
    assert '"raw_test_text_printed_or_persisted": False' in source
    assert '"qwen_or_nli_run": False' in source
