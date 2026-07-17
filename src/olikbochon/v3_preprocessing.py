"""Deterministic BanglaBERT preprocessing without model or network dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from .bangla_normalizer import normalize as official_normalize


PROMPT_MARKER = "[PROMPT]"
CONTEXT_PRESENT_MARKER = "[CONTEXT_PRESENT]"
CONTEXT_MARKER = "[CONTEXT]"
RESPONSE_MARKER = "[RESPONSE]"
STRUCTURAL_MARKERS = (
    PROMPT_MARKER,
    CONTEXT_PRESENT_MARKER,
    CONTEXT_MARKER,
    RESPONSE_MARKER,
)
MAX_ENCODED_LENGTH = 512
DEFAULT_RESPONSE_BUDGET = 384


@dataclass(frozen=True)
class PreparedPair:
    """Normalized, marker-delimited transformer input pair."""

    sequence_a: str
    sequence_b: str
    context_present: int


@dataclass(frozen=True)
class PairEncoding:
    """Unpadded tokenizer output plus safe aggregate truncation metadata."""

    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None
    pair_special_tokens: int
    response_original_tokens: int
    response_kept_tokens: int
    response_truncated: bool
    response_head_count: int
    response_tail_count: int


def _is_scalar_missing(value: Any) -> bool:
    if value is None:
        return True
    value_type = type(value)
    if value_type.__module__.startswith("pandas.") and value_type.__name__ in {
        "NAType",
        "NaTType",
    }:
        return True
    if isinstance(value, str):
        return False
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def raw_context_is_present(value: Any) -> bool:
    """Determine context presence before string conversion or normalization."""
    if _is_scalar_missing(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(str(value).strip())


def _required_raw_text(value: Any, *, field_name: str) -> str:
    if _is_scalar_missing(value):
        raise ValueError(f"{field_name} must not be missing")
    return str(value)


def build_transformer_pair(
    prompt: Any,
    context: Any,
    response: Any,
    *,
    normalizer: Callable[[str], str] = official_normalize,
) -> PreparedPair:
    """Normalize raw fields separately, then insert ordinary literal markers."""
    present = int(raw_context_is_present(context))
    raw_context = str(context) if present else ""
    normalized_prompt = normalizer(_required_raw_text(prompt, field_name="prompt_bn"))
    normalized_context = normalizer(raw_context)
    normalized_response = normalizer(_required_raw_text(response, field_name="response_bn"))

    sequence_a = (
        f"{PROMPT_MARKER}\n{normalized_prompt}\n\n"
        f"{CONTEXT_PRESENT_MARKER}\n{present}\n\n"
        f"{CONTEXT_MARKER}\n{normalized_context}"
    )
    sequence_b = f"{RESPONSE_MARKER}\n{normalized_response}"
    return PreparedPair(sequence_a, sequence_b, present)


def pair_special_token_count(tokenizer: Any) -> int:
    """Return and validate the tokenizer's pair-special-token requirement."""
    count = tokenizer.num_special_tokens_to_add(pair=True)
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"Invalid pair special-token count: {count!r}")
    return count


def _as_int_list(values: Any, *, field_name: str) -> list[int]:
    result = [int(value) for value in values]
    if not result and field_name == "input_ids":
        raise ValueError("Tokenizer returned empty input_ids")
    return result


def encode_pair_with_response_fallback(
    tokenizer: Any,
    pair: PreparedPair,
    *,
    max_length: int = MAX_ENCODED_LENGTH,
    response_budget: int = DEFAULT_RESPONSE_BUDGET,
) -> PairEncoding:
    """Encode a pair using only-first truncation and a deterministic response fallback."""
    special_tokens = pair_special_token_count(tokenizer)
    raw_budget = max_length - special_tokens
    if raw_budget <= 0:
        raise ValueError("max_length leaves no room after pair special tokens")
    safe_response_budget = min(response_budget, raw_budget)
    if safe_response_budget <= 0:
        raise ValueError("response_budget must be positive")

    response_tokens = list(tokenizer.encode(pair.sequence_b, add_special_tokens=False))
    response_original_tokens = len(response_tokens)
    response_truncated = response_original_tokens > safe_response_budget
    head_count = 0
    tail_count = 0

    if response_truncated:
        head_count = (safe_response_budget + 1) // 2
        tail_count = safe_response_budget // 2
        kept_response = response_tokens[:head_count]
        if tail_count:
            kept_response += response_tokens[-tail_count:]
        sequence_a_tokens = list(tokenizer.encode(pair.sequence_a, add_special_tokens=False))
        encoded = tokenizer.prepare_for_model(
            sequence_a_tokens,
            pair_ids=kept_response,
            add_special_tokens=True,
            padding=False,
            truncation="only_first",
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
    else:
        kept_response = response_tokens
        encoded = tokenizer(
            pair.sequence_a,
            pair.sequence_b,
            add_special_tokens=True,
            padding=False,
            truncation="only_first",
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

    input_ids = _as_int_list(encoded["input_ids"], field_name="input_ids")
    attention_mask = _as_int_list(encoded["attention_mask"], field_name="attention_mask")
    raw_token_types = encoded.get("token_type_ids")
    token_type_ids = (
        _as_int_list(raw_token_types, field_name="token_type_ids")
        if raw_token_types is not None
        else None
    )
    if len(input_ids) > max_length:
        raise ValueError(f"Encoded pair has {len(input_ids)} tokens, exceeding {max_length}")
    if len(attention_mask) != len(input_ids):
        raise ValueError("attention_mask length does not match input_ids")
    if token_type_ids is not None and len(token_type_ids) != len(input_ids):
        raise ValueError("token_type_ids length does not match input_ids")

    return PairEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        pair_special_tokens=special_tokens,
        response_original_tokens=response_original_tokens,
        response_kept_tokens=len(kept_response),
        response_truncated=response_truncated,
        response_head_count=head_count,
        response_tail_count=tail_count,
    )
