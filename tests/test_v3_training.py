from __future__ import annotations

from collections import UserDict
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from olikbochon.v3_training import (
    DynamicPairCollator,
    EXPECTED_BASE_MISSING,
    EXPECTED_BASE_UNEXPECTED,
    OPTIONAL_POSITION_IDS_KEY,
    TrainConfig,
    _validate_loading_info,
    _validate_synthetic_forward,
    expected_optimizer_steps,
    place_oof_probabilities,
    run_with_oom_restart,
    stage_a_rank,
)


class SyntheticOOM(RuntimeError):
    pass


def loading_info(*, missing: set[str], unexpected: set[str]) -> dict[str, object]:
    return {
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        "mismatched_keys": [],
        "error_msgs": [],
    }


def test_base_transition_without_optional_position_ids_key() -> None:
    appeared = _validate_loading_info(
        loading_info(missing=EXPECTED_BASE_MISSING, unexpected=EXPECTED_BASE_UNEXPECTED),
        expected_missing=EXPECTED_BASE_MISSING,
        expected_unexpected=EXPECTED_BASE_UNEXPECTED,
    )
    assert not appeared


def test_base_transition_with_exact_optional_position_ids_key() -> None:
    appeared = _validate_loading_info(
        loading_info(
            missing=EXPECTED_BASE_MISSING,
            unexpected=EXPECTED_BASE_UNEXPECTED | {OPTIONAL_POSITION_IDS_KEY},
        ),
        expected_missing=EXPECTED_BASE_MISSING,
        expected_unexpected=EXPECTED_BASE_UNEXPECTED,
    )
    assert appeared


def test_base_transition_rejects_another_unexpected_electra_key() -> None:
    with pytest.raises(RuntimeError, match="Unexplained"):
        _validate_loading_info(
            loading_info(
                missing=EXPECTED_BASE_MISSING,
                unexpected=EXPECTED_BASE_UNEXPECTED | {"electra.embeddings.token_type_ids"},
            ),
            expected_missing=EXPECTED_BASE_MISSING,
            expected_unexpected=EXPECTED_BASE_UNEXPECTED,
        )


def test_base_transition_rejects_missing_trainable_encoder_parameter() -> None:
    with pytest.raises(RuntimeError, match="Missing reusable ELECTRA"):
        _validate_loading_info(
            loading_info(
                missing=EXPECTED_BASE_MISSING | {"electra.encoder.layer.0.weight"},
                unexpected=EXPECTED_BASE_UNEXPECTED,
            ),
            expected_missing=EXPECTED_BASE_MISSING,
            expected_unexpected=EXPECTED_BASE_UNEXPECTED,
        )


def test_downstream_checkpoint_reload_is_clean() -> None:
    appeared = _validate_loading_info(
        loading_info(missing=set(), unexpected=set()),
        expected_missing=set(),
        expected_unexpected=set(),
    )
    assert not appeared


def test_downstream_reload_allows_only_optional_position_ids_buffer() -> None:
    appeared = _validate_loading_info(
        loading_info(missing=set(), unexpected={OPTIONAL_POSITION_IDS_KEY}),
        expected_missing=set(),
        expected_unexpected=set(),
    )
    assert appeared


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class FakeFiniteResult:
    def all(self) -> FakeFiniteResult:
        return self

    def item(self) -> bool:
        return True


class FakeTorch:
    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()

    @staticmethod
    def isfinite(_value: FakeTensor) -> FakeFiniteResult:
        return FakeFiniteResult()


class FakeTokenizer:
    vocab_size = 32_000

    def __init__(self) -> None:
        self.called = False

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(self, *_texts: str, **kwargs: object) -> UserDict[str, FakeTensor]:
        self.called = True
        assert kwargs == {
            "truncation": True,
            "max_length": 16,
            "return_tensors": "pt",
        }
        return UserDict(
            {
                "input_ids": FakeTensor((1, 7)),
                "attention_mask": FakeTensor((1, 7)),
            }
        )


class FakeClassifier:
    def __init__(self) -> None:
        self.config = SimpleNamespace(vocab_size=32_000, max_position_embeddings=512)
        self.eval_called = False
        self.forward_called = False

    def named_parameters(self) -> list[tuple[str, object]]:
        return [
            ("electra.embeddings.word_embeddings.weight", object()),
            ("classifier.out_proj.weight", object()),
        ]

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **inputs: FakeTensor) -> SimpleNamespace:
        self.forward_called = True
        assert set(inputs) == {"input_ids", "attention_mask"}
        return SimpleNamespace(logits=FakeTensor((1, 2)))


def test_synthetic_forward_validates_compatible_model() -> None:
    model = FakeClassifier()
    tokenizer = FakeTokenizer()
    result = _validate_synthetic_forward(model, tokenizer, torch_module=FakeTorch)
    assert result == {"synthetic_sequence_length": 7, "max_position_embeddings": 512}
    assert tokenizer.called
    assert model.eval_called
    assert model.forward_called


def test_optional_position_ids_key_must_not_be_trainable() -> None:
    model = FakeClassifier()
    model.named_parameters = lambda: [(OPTIONAL_POSITION_IDS_KEY, object())]  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="unexpectedly became trainable"):
        _validate_synthetic_forward(model, FakeTokenizer(), torch_module=FakeTorch)


def test_loading_info_still_rejects_mismatches_and_errors() -> None:
    with pytest.raises(RuntimeError, match="mismatch/errors"):
        _validate_loading_info(
            {
                "missing_keys": sorted(EXPECTED_BASE_MISSING),
                "unexpected_keys": sorted(EXPECTED_BASE_UNEXPECTED),
                "mismatched_keys": [
                    ("electra.embeddings.word_embeddings.weight", (1,), (2,))
                ],
                "error_msgs": [],
            },
            expected_missing=EXPECTED_BASE_MISSING,
            expected_unexpected=EXPECTED_BASE_UNEXPECTED,
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
