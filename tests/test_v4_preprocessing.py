from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from olikbochon.v3_preprocessing import build_transformer_pair, raw_context_is_present
from olikbochon.v4_preprocessing import (
    ROUTED,
    STRUCTURED,
    V3_COMPATIBLE,
    FieldBudget,
    encode_field_aware,
    official_context_is_present,
    prepare_v4_input,
    truncation_statistics,
)
from olikbochon.v4_runner import context_presence


class CharacterTokenizer:
    """Small reversible tokenizer fixture with BERT-like pair framing."""

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = [ord(character) + 1000 for character in text]
        return [101, *values, 102] if add_special_tokens else values

    @staticmethod
    def num_special_tokens_to_add(*, pair: bool) -> int:
        return 3 if pair else 2

    @staticmethod
    def prepare_for_model(
        first: list[int],
        *,
        pair_ids: list[int],
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool,
        return_token_type_ids: bool,
    ) -> dict[str, list[int]]:
        assert add_special_tokens and not padding and not truncation
        assert return_attention_mask and return_token_type_ids
        input_ids = [101, *first, 102, *pair_ids, 102]
        first_length = len(first) + 2
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": [0] * first_length + [1] * (len(pair_ids) + 1),
        }

    def __call__(
        self,
        first: str,
        second: str,
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        assert kwargs["truncation"] == "only_first"
        maximum = int(kwargs["max_length"])
        first_ids = self.encode(first, add_special_tokens=False)
        second_ids = self.encode(second, add_special_tokens=False)
        first_limit = maximum - len(second_ids) - 3
        return self.prepare_for_model(
            first_ids[: max(first_limit, 0)],
            pair_ids=second_ids,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )


def contains(values: tuple[int, ...], subsequence: list[int]) -> bool:
    return any(
        list(values[index : index + len(subsequence)]) == subsequence
        for index in range(len(values) - len(subsequence) + 1)
    )


def test_v3_compatible_serialization_parity() -> None:
    v3 = build_transformer_pair("question", "evidence", "answer")
    v4 = prepare_v4_input(
        "question", "evidence", "answer", serialization=V3_COMPATIBLE
    )
    assert (v4.sequence_a, v4.sequence_b, int(v4.context_present)) == (
        v3.sequence_a,
        v3.sequence_b,
        v3.context_present,
    )


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, "", " \t\n"])
def test_v3_and_v4_agree_on_ordinary_missing_context(value: Any) -> None:
    assert not raw_context_is_present(value)
    assert not official_context_is_present(value)


@pytest.mark.parametrize(
    "value",
    ["[NULL]", "[null]", " [NuLl] ", "NULL", "none", "N/A", " n/a "],
)
def test_v4_treats_normalized_official_sentinels_as_absent(value: str) -> None:
    assert raw_context_is_present(value)
    assert not official_context_is_present(value)
    prepared = prepare_v4_input("question", value, "answer", serialization=V3_COMPATIBLE)
    assert not prepared.context_present
    assert "[CONTEXT_PRESENT]\n0" in prepared.sequence_a
    assert prepared.context == ""


def test_v4_detects_sentinel_after_unicode_normalization() -> None:
    assert not official_context_is_present("\uff3b\uff2e\uff35\uff2c\uff2c\uff3d")


def test_v3_and_v4_agree_on_ordinary_bengali_context() -> None:
    context = "\u098f\u099f\u09bf \u098f\u0995\u099f\u09bf \u09b8\u09be\u09a7\u09be\u09b0\u09a3 \u09aa\u09cd\u09b0\u09b8\u0999\u09cd\u0997\u0964"
    assert raw_context_is_present(context)
    assert official_context_is_present(context)
    assert prepare_v4_input(
        "question", context, "answer", serialization=ROUTED
    ).context_present


def test_runner_and_v4_preparation_use_the_same_route_policy() -> None:
    values = [None, np.nan, "", " [NULL] ", "NULL", "evidence"]
    frame = pd.DataFrame({"context": values})
    expected = tuple(official_context_is_present(value) for value in frame["context"])
    assert context_presence(frame) == expected == (False, False, False, False, False, True)


def test_structured_serialization_uses_plain_markers_and_null() -> None:
    prepared = prepare_v4_input("question", None, "answer", serialization=STRUCTURED)
    assert prepared.sequence_a == "[QUESTION]\nquestion\n\n[CONTEXT]\n[NULL]"
    assert prepared.sequence_b == "[ANSWER]\nanswer"
    assert not prepared.context_present


def test_routed_null_context_has_no_context_block() -> None:
    prepared = prepare_v4_input("question", "", "answer", serialization=ROUTED)
    assert prepared.sequence_a == "[QUESTION]\nquestion"
    assert "CONTEXT" not in prepared.sequence_a
    assert "EVIDENCE" not in prepared.sequence_a
    assert prepared.sequence_b == "[CLAIM]\nanswer"


def test_short_prompt_and_response_are_preserved() -> None:
    tokenizer = CharacterTokenizer()
    prepared = prepare_v4_input(
        "important question",
        "x" * 500,
        "short answer",
        serialization=ROUTED,
    )
    encoded = encode_field_aware(
        tokenizer,
        prepared,
        budget=FieldBudget(maximum_length=256, prompt_tokens=40, response_tokens=40),
    )
    assert contains(encoded.input_ids, tokenizer.encode(prepared.prompt))
    assert contains(encoded.input_ids, tokenizer.encode(prepared.response))
    assert len(encoded.input_ids) <= 256


def test_long_context_uses_deterministic_head_plus_tail() -> None:
    tokenizer = CharacterTokenizer()
    context = "0123456789" * 80
    prepared = prepare_v4_input("q", context, "a", serialization=ROUTED)
    budget = FieldBudget(
        maximum_length=128,
        prompt_tokens=8,
        response_tokens=8,
        context_head_ratio=0.5,
    )
    first = encode_field_aware(tokenizer, prepared, budget=budget)
    second = encode_field_aware(tokenizer, prepared, budget=budget)
    assert first == second
    assert first.lengths.context_retained < first.lengths.context_original
    retained = first.lengths.context_retained
    head = round(retained * budget.context_head_ratio)
    tail = retained - head
    context_ids = tokenizer.encode(context)
    assert contains(first.input_ids, context_ids[:head])
    assert contains(first.input_ids, context_ids[-tail:])


def test_train_and_inference_preprocessing_are_the_same_pure_call() -> None:
    tokenizer = CharacterTokenizer()
    prepared = prepare_v4_input("q", "context", "a", serialization=STRUCTURED)
    budget = FieldBudget(maximum_length=128, prompt_tokens=20, response_tokens=20)
    train_encoding = encode_field_aware(tokenizer, prepared, budget=budget)
    inference_encoding = encode_field_aware(tokenizer, prepared, budget=budget)
    assert train_encoding == inference_encoding


def test_truncation_summary_contains_aggregate_fields_only() -> None:
    tokenizer = CharacterTokenizer()
    rows = [
        encode_field_aware(
            tokenizer,
            prepare_v4_input("q", None, "a", serialization=STRUCTURED),
            budget=FieldBudget(maximum_length=128, prompt_tokens=20, response_tokens=20),
        ),
        encode_field_aware(
            tokenizer,
            prepare_v4_input("q", "c" * 300, "a", serialization=STRUCTURED),
            budget=FieldBudget(maximum_length=128, prompt_tokens=20, response_tokens=20),
        ),
    ]
    summary = truncation_statistics(rows)
    assert summary["null_context_count"] == 1
    assert summary["context_present_count"] == 1
    assert summary["context"]["truncated_count"] == 1
    forbidden = {"text", "texts", "raw_text", "input_ids", "rows_by_item"}
    assert forbidden.isdisjoint(summary)
    assert all(forbidden.isdisjoint(value) for value in summary.values() if isinstance(value, dict))
