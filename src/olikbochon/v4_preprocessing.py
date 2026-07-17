"""Deterministic Version 4 serialization and field-aware token budgeting."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable

from .v3_default_normalizer import normalize_default
from .v3_preprocessing import build_transformer_pair, raw_context_is_present


V3_COMPATIBLE = "v3_compatible"
STRUCTURED = "structured"
ROUTED = "routed"
SERIALIZATIONS = (V3_COMPATIBLE, STRUCTURED, ROUTED)


@dataclass(frozen=True)
class PreparedV4Input:
    """Normalized pair plus its separately retained model fields."""

    serialization: str
    sequence_a: str
    sequence_b: str
    prompt: str
    context: str
    response: str
    context_present: bool


@dataclass(frozen=True)
class FieldBudget:
    """Frozen maximum token allocation before pair special tokens."""

    maximum_length: int = 256
    prompt_tokens: int = 80
    response_tokens: int = 96
    context_head_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.maximum_length <= 0 or self.prompt_tokens <= 0 or self.response_tokens <= 0:
            raise ValueError("Token lengths and field budgets must be positive")
        if not 0.0 <= self.context_head_ratio <= 1.0:
            raise ValueError("context_head_ratio must be between zero and one")


@dataclass(frozen=True)
class FieldLengths:
    """Aggregate-safe original and retained field lengths for one input."""

    prompt_original: int
    prompt_retained: int
    context_original: int
    context_retained: int
    response_original: int
    response_retained: int


@dataclass(frozen=True)
class EncodedV4Input:
    """Unpadded model input and aggregate-only truncation metadata."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None
    lengths: FieldLengths
    context_present: bool
    serialization: str


def _required_text(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must not be missing")
    return normalize_default(str(value))


def prepare_v4_input(
    prompt: Any,
    context: Any,
    response: Any,
    *,
    serialization: str,
) -> PreparedV4Input:
    """Build one frozen ordinary-text serialization without tokenizer mutation."""
    if serialization not in SERIALIZATIONS:
        raise ValueError(f"Unknown serialization {serialization!r}; expected {SERIALIZATIONS}")
    normalized_prompt = _required_text(prompt, "prompt_bn")
    normalized_response = _required_text(response, "response_bn")
    present = raw_context_is_present(context)
    normalized_context = normalize_default(str(context)) if present else ""

    if serialization == V3_COMPATIBLE:
        pair = build_transformer_pair(prompt, context, response)
        return PreparedV4Input(
            serialization,
            pair.sequence_a,
            pair.sequence_b,
            normalized_prompt,
            normalized_context,
            normalized_response,
            present,
        )
    if serialization == STRUCTURED:
        shown_context = normalized_context if present else "[NULL]"
        sequence_a = (
            f"[QUESTION]\n{normalized_prompt}\n\n[CONTEXT]\n{shown_context}"
        )
        sequence_b = f"[ANSWER]\n{normalized_response}"
    elif present:
        sequence_a = (
            f"[QUESTION]\n{normalized_prompt}\n\n[EVIDENCE]\n{normalized_context}"
        )
        sequence_b = f"[CLAIM]\n{normalized_response}"
    else:
        sequence_a = f"[QUESTION]\n{normalized_prompt}"
        sequence_b = f"[CLAIM]\n{normalized_response}"
    return PreparedV4Input(
        serialization,
        sequence_a,
        sequence_b,
        normalized_prompt,
        normalized_context,
        normalized_response,
        present,
    )


def _tokens(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def _head_tail(values: list[int], limit: int, head_ratio: float) -> list[int]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return list(values)
    head = round(limit * head_ratio)
    head = min(max(head, 0), limit)
    tail = limit - head
    return values[:head] + (values[-tail:] if tail else [])


def _layout(prepared: PreparedV4Input) -> tuple[str, str, str]:
    """Return prompt prefix, context prefix, and response prefix."""
    if prepared.serialization == V3_COMPATIBLE:
        context_prefix = (
            f"\n\n[CONTEXT_PRESENT]\n{int(prepared.context_present)}"
            "\n\n[CONTEXT]\n"
        )
        return "[PROMPT]\n", context_prefix, "[RESPONSE]\n"
    if prepared.serialization == STRUCTURED:
        context_prefix = "\n\n[CONTEXT]\n" if prepared.context_present else "\n\n[CONTEXT]\n[NULL]"
        return "[QUESTION]\n", context_prefix, "[ANSWER]\n"
    if prepared.context_present:
        return "[QUESTION]\n", "\n\n[EVIDENCE]\n", "[CLAIM]\n"
    return "[QUESTION]\n", "", "[CLAIM]\n"


def encode_field_aware(
    tokenizer: Any,
    prepared: PreparedV4Input,
    *,
    budget: FieldBudget = FieldBudget(),
) -> EncodedV4Input:
    """Encode fields under frozen budgets, assigning all residual capacity to context."""
    prompt_prefix, context_prefix, response_prefix = _layout(prepared)
    prompt_prefix_ids = _tokens(tokenizer, prompt_prefix)
    context_prefix_ids = _tokens(tokenizer, context_prefix)
    response_prefix_ids = _tokens(tokenizer, response_prefix)
    marker_count = len(prompt_prefix_ids) + len(context_prefix_ids) + len(response_prefix_ids)
    pair_special = int(tokenizer.num_special_tokens_to_add(pair=True))
    capacity = budget.maximum_length - pair_special - marker_count
    if capacity < budget.prompt_tokens + budget.response_tokens:
        raise ValueError(
            "maximum_length cannot fit the frozen prompt and response budgets after markers"
        )

    prompt = _tokens(tokenizer, prepared.prompt)
    context = _tokens(tokenizer, prepared.context) if prepared.context_present else []
    response = _tokens(tokenizer, prepared.response)
    kept_prompt = _head_tail(prompt, budget.prompt_tokens, 0.5)
    kept_response = _head_tail(response, budget.response_tokens, 0.5)
    context_capacity = capacity - len(kept_prompt) - len(kept_response)
    kept_context = _head_tail(context, context_capacity, budget.context_head_ratio)

    sequence_a = prompt_prefix_ids + kept_prompt + context_prefix_ids + kept_context
    sequence_b = response_prefix_ids + kept_response
    encoded = tokenizer.prepare_for_model(
        sequence_a,
        pair_ids=sequence_b,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids = tuple(int(value) for value in encoded["input_ids"])
    if len(input_ids) > budget.maximum_length:
        raise RuntimeError("Field-aware encoding exceeded maximum_length")
    token_types = encoded.get("token_type_ids")
    return EncodedV4Input(
        input_ids,
        tuple(int(value) for value in encoded["attention_mask"]),
        tuple(int(value) for value in token_types) if token_types is not None else None,
        FieldLengths(
            len(prompt),
            len(kept_prompt),
            len(context),
            len(kept_context),
            len(response),
            len(kept_response),
        ),
        prepared.context_present,
        prepared.serialization,
    )


def encode_comparison_baseline(
    tokenizer: Any,
    prepared: PreparedV4Input,
    *,
    maximum_length: int = 256,
) -> EncodedV4Input:
    """Encode A-C with identical pair-level response-protected truncation."""
    encoded = tokenizer(
        prepared.sequence_a,
        prepared.sequence_b,
        add_special_tokens=True,
        padding=False,
        truncation="only_first",
        max_length=maximum_length,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    prompt = _tokens(tokenizer, prepared.prompt)
    context = _tokens(tokenizer, prepared.context) if prepared.context_present else []
    response = _tokens(tokenizer, prepared.response)
    token_types = encoded.get("token_type_ids")
    return EncodedV4Input(
        tuple(int(value) for value in encoded["input_ids"]),
        tuple(int(value) for value in encoded["attention_mask"]),
        tuple(int(value) for value in token_types) if token_types is not None else None,
        FieldLengths(len(prompt), len(prompt), len(context), len(context), len(response), len(response)),
        prepared.context_present,
        prepared.serialization,
    )


def truncation_statistics(encodings: Iterable[EncodedV4Input]) -> dict[str, Any]:
    """Summarize field retention without exposing row-level values or text."""
    rows = list(encodings)
    if not rows:
        raise ValueError("At least one encoding is required")
    result: dict[str, Any] = {
        "rows": len(rows),
        "context_present_count": sum(row.context_present for row in rows),
        "null_context_count": sum(not row.context_present for row in rows),
    }
    for field in ("prompt", "context", "response"):
        original = [getattr(row.lengths, f"{field}_original") for row in rows]
        retained = [getattr(row.lengths, f"{field}_retained") for row in rows]
        truncated = sum(after < before for before, after in zip(original, retained))
        result[field] = {
            "truncated_count": truncated,
            "truncated_percent": 100.0 * truncated / len(rows),
            "mean_original_tokens": fmean(original),
            "mean_retained_tokens": fmean(retained),
        }
    return result
