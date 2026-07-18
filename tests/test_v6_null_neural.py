from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from olikbochon.v5_lexical import deterministic_substring_prediction
from olikbochon.v6_null_neural import (
    CANDIDATE_J,
    CANDIDATE_K,
    CANDIDATE_L,
    HEAD_LEARNING_RATE,
    MAXIMUM_LENGTH,
    ROUTED_CHAMPION,
    TOP_LAYER_LEARNING_RATE,
    _load_delta_checkpoint,
    _save_delta_checkpoint,
    configure_candidate,
    encode_null_frame,
    optimizer_groups,
    require_absent_frame,
    serialize_null_input,
    training_class_weights,
)


def absent_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context": ["[NULL]", None],
            "prompt_bn": ["  প্রশ্ন এক  ", "প্রশ্ন দুই"],
            "response_bn": ["উত্তর এক", "উত্তর দুই"],
            "label": [0, 1],
        }
    )


class RecordingTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None = None,
        padding: bool | None = None,
    ) -> dict[str, list[int]]:
        assert add_special_tokens
        self.texts.append(text)
        length = len(text) + 2
        if truncation:
            assert max_length == MAXIMUM_LENGTH
            length = min(length, max_length)
            assert padding is False
        return {
            "input_ids": list(range(length)),
            "attention_mask": [1] * length,
            "token_type_ids": [0] * length,
        }


def test_null_serialization_is_exact_and_never_uses_context() -> None:
    assert serialize_null_input("  প্রশ্ন  ", "  উত্তর  ") == (
        "[QUESTION]\nপ্রশ্ন\n\n[ANSWER]\nউত্তর"
    )
    tokenizer = RecordingTokenizer()
    first, first_audit = encode_null_frame(absent_frame(), tokenizer)
    second, second_audit = encode_null_frame(absent_frame(), RecordingTokenizer())
    assert first == second
    assert first_audit == second_audit
    assert first.context_present == (False, False)
    assert all("[NULL]" not in text for text in tokenizer.texts)
    assert all("[CONTEXT]" not in text for text in tokenizer.texts)
    assert first_audit["maximum_length"] == 192


def test_null_encoder_rejects_context_present_rows_and_length_override() -> None:
    frame = absent_frame()
    frame.loc[0, "context"] = "সাধারণ প্রমাণ"
    with pytest.raises(ValueError, match="context-absent"):
        require_absent_frame(frame)
    with pytest.raises(ValueError, match="frozen at 192"):
        encode_null_frame(absent_frame(), RecordingTokenizer(), maximum_length=256)


def test_class_weights_use_only_supplied_training_labels() -> None:
    weights = training_class_weights([0, 0, 0, 1])
    assert np.allclose(weights, [4 / 6, 2.0])
    assert np.array_equal(weights, training_class_weights([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="both labels"):
        training_class_weights([0, 0, 0])


def test_present_route_remains_the_frozen_substring_rule() -> None:
    assert ROUTED_CHAMPION == 0.692051
    assert deterministic_substring_prediction("ঢাকা ২০২৬", "ঢাকা 2026") == 1
    assert deterministic_substring_prediction("ঢাকা", "চট্টগ্রাম") == 0


def fake_model():
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(4)])

    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = torch.nn.Embedding(20, 4)
            self.encoder = Encoder()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.electra = Base()
            self.classifier = torch.nn.Linear(4, 2)
            self.config = SimpleNamespace(hidden_size=4, hidden_dropout_prob=0.1)

        @property
        def base_model(self):
            return self.electra

    return Model()


def test_candidate_j_freezes_encoder_and_trains_only_small_linear_head() -> None:
    model = fake_model()
    audit = configure_candidate(model, CANDIDATE_J)
    assert audit["trainable_parameters"] == 10
    assert all(name.startswith("classifier.") for name in audit["trainable_parameter_names"])
    assert all(not parameter.requires_grad for parameter in model.electra.parameters())
    groups = optimizer_groups(model, CANDIDATE_J)
    assert len(groups) == 1 and groups[0]["lr"] == HEAD_LEARNING_RATE


def test_candidate_k_trains_exactly_top_two_layers_and_head() -> None:
    model = fake_model()
    audit = configure_candidate(model, CANDIDATE_K)
    assert all(not parameter.requires_grad for parameter in model.electra.embeddings.parameters())
    assert all(
        not parameter.requires_grad
        for layer in model.electra.encoder.layer[:2]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.electra.encoder.layer[-2:]
        for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert audit["trainable_parameters"] == 50
    assert [group["lr"] for group in optimizer_groups(model, CANDIDATE_K)] == [
        HEAD_LEARNING_RATE,
        TOP_LAYER_LEARNING_RATE,
    ]


def test_candidate_l_full_finetune_has_no_frozen_parameters() -> None:
    model = fake_model()
    audit = configure_candidate(model, CANDIDATE_L)
    assert audit["trainable_parameters"] == audit["total_parameters"]
    assert audit["frozen_parameters"] == 0
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_delta_checkpoint_reloads_only_frozen_candidate_state_exactly(
    tmp_path, monkeypatch
) -> None:
    import torch
    import olikbochon.v6_null_neural as module

    torch.manual_seed(17)
    model = fake_model()
    configure_candidate(model, CANDIDATE_J)
    with torch.no_grad():
        for parameter in model.classifier.parameters():
            parameter.fill_(0.25)
    checkpoint = tmp_path / "checkpoint"
    _save_delta_checkpoint(
        model,
        checkpoint,
        candidate=CANDIDATE_J,
        seed=17,
        selected_epoch=2,
    )
    monkeypatch.setattr(
        module,
        "load_offline_base",
        lambda _path: (object(), fake_model(), {}),
    )
    _, reloaded = _load_delta_checkpoint(tmp_path, checkpoint, CANDIDATE_J)
    assert all(
        torch.equal(parameter, torch.full_like(parameter, 0.25))
        for parameter in reloaded.classifier.parameters()
    )
    metadata = (checkpoint / "checkpoint.json").read_text(encoding="utf-8")
    assert "trainable_parameters_only" in metadata
    assert "প্রশ্ন" not in metadata and "উত্তর" not in metadata
