"""Safe label and vocabulary helpers for the Version 3 transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FAITHFUL_LABEL = "FAITHFUL"


def _coerce_index(value: object, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} contains boolean index {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    raise ValueError(f"{source} contains non-integer index {value!r}")


def _normalized_id2label(mapping: Mapping[object, object]) -> dict[int, str]:
    normalized: dict[int, str] = {}
    for raw_index, raw_label in mapping.items():
        index = _coerce_index(raw_index, source="id2label")
        if index in normalized:
            raise ValueError(f"id2label resolves multiple entries to index {index}")
        if not isinstance(raw_label, str) or not raw_label:
            raise ValueError(f"id2label[{raw_index!r}] is not a nonempty label")
        normalized[index] = raw_label
    return normalized


def _normalized_label2id(mapping: Mapping[object, object]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    used_indices: set[int] = set()
    for raw_label, raw_index in mapping.items():
        if not isinstance(raw_label, str) or not raw_label:
            raise ValueError(f"label2id contains invalid label {raw_label!r}")
        index = _coerce_index(raw_index, source="label2id")
        if raw_label in normalized:
            raise ValueError(f"label2id contains duplicate label {raw_label!r}")
        if index in used_indices:
            raise ValueError(f"label2id resolves multiple labels to index {index}")
        normalized[raw_label] = index
        used_indices.add(index)
    return normalized


def resolve_faithful_logit_index(config: Any, *, logits_dimension: int) -> int:
    """Resolve the FAITHFUL logit index from mutually consistent config mappings."""
    if logits_dimension <= 0:
        raise ValueError("logits_dimension must be positive")

    raw_label2id = getattr(config, "label2id", None)
    raw_id2label = getattr(config, "id2label", None)
    if not isinstance(raw_label2id, Mapping) or not raw_label2id:
        raise ValueError("model.config.label2id is missing or empty")
    if not isinstance(raw_id2label, Mapping) or not raw_id2label:
        raise ValueError("model.config.id2label is missing or empty")

    label2id = _normalized_label2id(raw_label2id)
    id2label = _normalized_id2label(raw_id2label)
    if set(label2id.values()) != set(id2label):
        raise ValueError("label2id and id2label expose different index sets")
    for label, index in label2id.items():
        if id2label.get(index) != label:
            raise ValueError(f"label mappings disagree for {label!r} at index {index}")

    if FAITHFUL_LABEL not in label2id:
        raise ValueError("FAITHFUL is absent from model.config.label2id")
    faithful_indices = [index for index, label in id2label.items() if label == FAITHFUL_LABEL]
    if len(faithful_indices) != 1:
        raise ValueError("FAITHFUL must resolve to exactly one id2label index")

    index = label2id[FAITHFUL_LABEL]
    if faithful_indices[0] != index:
        raise ValueError("FAITHFUL mappings disagree")
    if not 0 <= index < logits_dimension:
        raise ValueError(
            f"FAITHFUL index {index} is outside logits dimension {logits_dimension}"
        )
    return index


def validate_tokenizer_model_vocabulary(tokenizer: Any, model_config: Any) -> int:
    """Require tokenizer length, reported vocabulary, and model vocabulary to agree."""
    tokenizer_length = len(tokenizer)
    tokenizer_vocab = getattr(tokenizer, "vocab_size", None)
    model_vocab = getattr(model_config, "vocab_size", None)
    if not isinstance(tokenizer_vocab, int) or not isinstance(model_vocab, int):
        raise ValueError("tokenizer and model must expose integer vocab_size values")
    if tokenizer_length != tokenizer_vocab or tokenizer_vocab != model_vocab:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"len(tokenizer)={tokenizer_length}, tokenizer.vocab_size={tokenizer_vocab}, "
            f"model.config.vocab_size={model_vocab}"
        )
    return model_vocab
