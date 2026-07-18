from __future__ import annotations

import inspect

from olikbochon import v12_runner


def test_submission_is_strictly_gated_and_has_safe_score_columns() -> None:
    source = inspect.getsource(v12_runner.main)
    assert source.index("if gate_passed:") < source.index("submission.to_csv")
    for column in (
        '"id"',
        '"route"',
        '"exact_template_override"',
        '"qwen_label"',
        '"qwen_confidence"',
        '"predicted_label"',
    ):
        assert column in source
    assert "prompt_bn\":" not in source
    assert "response_bn\":" not in source
