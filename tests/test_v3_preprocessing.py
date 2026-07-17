from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from olikbochon.v3_preprocessing import (
    CONTEXT_MARKER,
    CONTEXT_PRESENT_MARKER,
    PROMPT_MARKER,
    RESPONSE_MARKER,
    PreparedPair,
    build_transformer_pair,
    encode_pair_with_response_fallback,
    pair_special_token_count,
    raw_context_is_present,
)


class SyntheticTokenizer:
    """Small deterministic pair tokenizer; token-addition methods are fatal."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self.last_pair_ids: list[int] | None = None

    def _ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in text.split():
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1000
            ids.append(self._vocab[token])
        return ids

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self._ids(text)

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair
        return 3

    def _prepare(
        self, first: list[int], second: list[int], *, max_length: int
    ) -> dict[str, list[int]]:
        first_budget = max_length - len(second) - 3
        if first_budget < 0:
            raise ValueError("second sequence cannot fit")
        first = first[:first_budget]
        input_ids = [101, *first, 102, *second, 102]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": [0] * (len(first) + 2) + [1] * (len(second) + 1),
        }

    def __call__(
        self,
        first: str,
        second: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: str,
        max_length: int,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens and not padding and truncation == "only_first"
        assert return_attention_mask and return_token_type_ids
        second_ids = self._ids(second)
        self.last_pair_ids = second_ids
        return self._prepare(self._ids(first), second_ids, max_length=max_length)

    def prepare_for_model(
        self,
        first: list[int],
        *,
        pair_ids: list[int],
        add_special_tokens: bool,
        padding: bool,
        truncation: str,
        max_length: int,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens and not padding and truncation == "only_first"
        assert return_attention_mask and return_token_type_ids
        self.last_pair_ids = list(pair_ids)
        return self._prepare(list(first), list(pair_ids), max_length=max_length)

    def add_tokens(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("tokenizer.add_tokens must never be called")

    def add_special_tokens(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("tokenizer.add_special_tokens must never be called")


@pytest.mark.parametrize("value", [None, pd.NA, float("nan"), "", " ", "\t\n"])
def test_raw_missing_context_cases(value: object) -> None:
    assert not raw_context_is_present(value)


def test_present_context_is_not_reclassified_by_old_sentinels() -> None:
    assert raw_context_is_present("[NULL]")
    assert raw_context_is_present("প্রাসঙ্গিক তথ্য")
    assert raw_context_is_present(0)


def test_presence_is_detected_before_conversion_and_normalization() -> None:
    calls: list[str] = []

    def recording_normalizer(value: str) -> str:
        calls.append(value)
        return f"normalized<{value}>"

    pair = build_transformer_pair("প্রশ্ন", float("nan"), "উত্তর", normalizer=recording_normalizer)
    assert pair.context_present == 0
    assert calls == ["প্রশ্ন", "", "উত্তর"]
    assert f"{CONTEXT_PRESENT_MARKER}\n0" in pair.sequence_a
    assert f"{CONTEXT_MARKER}\nnormalized<>" in pair.sequence_a


def test_official_nfkc_normalization_precedes_literal_markers() -> None:
    pair = build_transformer_pair("Ａ প্রশ্ন", "  তথ্য  ", "１２ উত্তর")
    assert pair.sequence_a.startswith(f"{PROMPT_MARKER}\nA প্রশ্ন")
    assert f"{CONTEXT_PRESENT_MARKER}\n1" in pair.sequence_a
    assert pair.sequence_b.startswith(f"{RESPONSE_MARKER}\n12 উত্তর")
    assert pair.sequence_a.count(PROMPT_MARKER) == 1
    assert pair.sequence_b.count(RESPONSE_MARKER) == 1


def test_pair_special_token_count() -> None:
    assert pair_special_token_count(SyntheticTokenizer()) == 3


def test_normal_only_first_path_preserves_fitting_response() -> None:
    tokenizer = SyntheticTokenizer()
    pair = PreparedPair(" ".join(f"a{index}" for index in range(30)), "b0 b1 b2", 1)
    encoded = encode_pair_with_response_fallback(
        tokenizer, pair, max_length=12, response_budget=5
    )
    assert not encoded.response_truncated
    assert encoded.response_original_tokens == encoded.response_kept_tokens == 3
    assert len(encoded.input_ids) == 12
    assert tokenizer.last_pair_ids == tokenizer.encode(pair.sequence_b, add_special_tokens=False)


@pytest.mark.parametrize(
    ("budget", "expected_head", "expected_tail"), [(6, 3, 3), (5, 3, 2)]
)
def test_head_tail_response_fallback_uses_locked_split(
    budget: int, expected_head: int, expected_tail: int
) -> None:
    tokenizer = SyntheticTokenizer()
    pair = PreparedPair("a0 a1 a2", " ".join(f"b{index}" for index in range(12)), 1)
    original = tokenizer.encode(pair.sequence_b, add_special_tokens=False)
    encoded = encode_pair_with_response_fallback(
        tokenizer, pair, max_length=12, response_budget=budget
    )
    assert encoded.response_truncated
    assert encoded.response_head_count == expected_head
    assert encoded.response_tail_count == expected_tail
    assert tokenizer.last_pair_ids == original[:expected_head] + original[-expected_tail:]
    assert encoded.response_kept_tokens == budget
    assert len(encoded.input_ids) <= 12


def test_both_sequences_long_are_deterministic_and_bounded() -> None:
    tokenizer = SyntheticTokenizer()
    pair = PreparedPair(
        " ".join(f"a{index}" for index in range(100)),
        " ".join(f"b{index}" for index in range(100)),
        1,
    )
    first = encode_pair_with_response_fallback(tokenizer, pair, max_length=20, response_budget=7)
    second = encode_pair_with_response_fallback(tokenizer, pair, max_length=20, response_budget=7)
    assert first == second
    assert len(first.input_ids) == 20
    assert first.response_head_count == 4
    assert first.response_tail_count == 3


def test_required_fields_reject_missing_values() -> None:
    with pytest.raises(ValueError, match="prompt_bn"):
        build_transformer_pair(None, "context", "response")
    with pytest.raises(ValueError, match="response_bn"):
        build_transformer_pair("prompt", "context", pd.NA)
