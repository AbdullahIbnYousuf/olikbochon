from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from olikbochon.v3_training import (
    DynamicPairCollator,
    EXPECTED_BASE_MISSING,
    EXPECTED_BASE_UNEXPECTED,
    TrainConfig,
    _validate_loading_info,
    expected_optimizer_steps,
    place_oof_probabilities,
    run_with_oom_restart,
    stage_a_rank,
)


class SyntheticOOM(RuntimeError):
    pass


def test_base_model_transition_accepts_only_expected_keys() -> None:
    _validate_loading_info(
        {
            "missing_keys": sorted(EXPECTED_BASE_MISSING),
            "unexpected_keys": sorted(EXPECTED_BASE_UNEXPECTED),
            "mismatched_keys": [],
            "error_msgs": [],
        },
        expected_missing=EXPECTED_BASE_MISSING,
        expected_unexpected=EXPECTED_BASE_UNEXPECTED,
    )
    with pytest.raises(RuntimeError, match="ELECTRA"):
        _validate_loading_info(
            {
                "missing_keys": ["electra.encoder.layer.0.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            expected_missing=set(),
            expected_unexpected=set(),
        )


def test_saved_checkpoint_requires_clean_loading_info() -> None:
    _validate_loading_info(
        {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []},
        expected_missing=set(),
        expected_unexpected=set(),
    )
    with pytest.raises(RuntimeError, match="Unexplained"):
        _validate_loading_info(
            {
                "missing_keys": ["classifier.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            expected_missing=set(),
            expected_unexpected=set(),
        )


def test_oom_restart_discards_partial_state_and_resets_everything() -> None:
    attempts: list[SimpleNamespace] = []
    cleanup_calls: list[int] = []
    seed_calls: list[int] = []

    def runner(config: TrainConfig, attempt: int) -> SimpleNamespace:
        state = SimpleNamespace(attempt=attempt, updates=0, config=config)
        attempts.append(state)
        state.updates = 3
        if attempt == 1:
            raise SyntheticOOM("CUDA out of memory")
        return state

    output, record = run_with_oom_restart(
        "synthetic_phase",
        TrainConfig(epochs=4, learning_rate=1e-5),
        runner,
        is_oom=lambda error: isinstance(error, SyntheticOOM),
        cleanup=lambda: cleanup_calls.append(1),
        seed_reset=lambda seed: seed_calls.append(seed),
    )
    assert len(attempts) == 2
    assert attempts[0] is not attempts[1]
    assert attempts[1].attempt == 2
    assert output is attempts[1]
    assert cleanup_calls == [1]
    assert seed_calls == [42]
    assert record.complete_restart
    assert record.final_batch_size == 4
    assert record.final_accumulation == 4
    assert record.final_effective_batch_size == 16


def test_oom_allows_only_one_restart() -> None:
    with pytest.raises(RuntimeError, match="exhausted"):
        run_with_oom_restart(
            "always_oom",
            TrainConfig(epochs=1, learning_rate=1e-5),
            lambda _config, _attempt: (_ for _ in ()).throw(SyntheticOOM()),
            is_oom=lambda error: isinstance(error, SyntheticOOM),
            cleanup=lambda: None,
            seed_reset=lambda _seed: None,
        )


def test_non_oom_error_is_not_retried() -> None:
    attempts = 0

    def fail(_config: TrainConfig, _attempt: int) -> None:
        nonlocal attempts
        attempts += 1
        raise ValueError("not OOM")

    with pytest.raises(ValueError, match="not OOM"):
        run_with_oom_restart(
            "failure",
            TrainConfig(epochs=1, learning_rate=1e-5),
            fail,
            is_oom=lambda _error: False,
            cleanup=lambda: None,
            seed_reset=lambda _seed: None,
        )
    assert attempts == 1


def test_gradient_accumulation_counts_final_incomplete_window() -> None:
    assert expected_optimizer_steps(batch_count=5, accumulation=2, epochs=4) == 12
    assert expected_optimizer_steps(batch_count=4, accumulation=2, epochs=4) == 8


def test_stage_a_checkpoint_tie_breaking() -> None:
    assert stage_a_rank(0.70, 0.5, 2) > stage_a_rank(0.69, 0.1, 1)
    assert stage_a_rank(0.70, 0.4, 2) > stage_a_rank(0.70, 0.5, 1)
    assert stage_a_rank(0.70, 0.4, 1) > stage_a_rank(0.70, 0.4, 2)


def test_oof_probabilities_are_placed_once_in_original_positions() -> None:
    destination = np.full(6, np.nan)
    place_oof_probabilities(destination, np.array([4, 1]), np.array([0.8, 0.2]))
    assert destination[4] == 0.8
    assert destination[1] == 0.2
    with pytest.raises(ValueError, match="already populated"):
        place_oof_probabilities(destination, np.array([1]), np.array([0.3]))


def test_dynamic_collator_delegates_padding_without_token_changes() -> None:
    class FakeTokenizer:
        def pad(self, features: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
            assert kwargs == {"padding": True, "return_tensors": "pt"}
            assert [len(item["input_ids"]) for item in features] == [2, 3]  # type: ignore[arg-type]
            return {"padded": True}

    collator = DynamicPairCollator(FakeTokenizer())
    result = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
        ]
    )
    assert result == {"padded": True}
