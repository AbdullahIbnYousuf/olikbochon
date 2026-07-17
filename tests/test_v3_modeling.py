from types import SimpleNamespace

import pytest

from olikbochon.v3_modeling import (
    resolve_faithful_logit_index,
    validate_tokenizer_model_vocabulary,
)


def config(label2id: object, id2label: object) -> SimpleNamespace:
    return SimpleNamespace(label2id=label2id, id2label=id2label)


def test_resolves_faithful_index_without_column_assumption() -> None:
    reversed_config = config(
        {"FAITHFUL": 0, "HALLUCINATED": 1},
        {"0": "FAITHFUL", "1": "HALLUCINATED"},
    )
    assert resolve_faithful_logit_index(reversed_config, logits_dimension=2) == 0


@pytest.mark.parametrize(
    ("label2id", "id2label", "message"),
    [
        (None, {0: "HALLUCINATED", 1: "FAITHFUL"}, "label2id"),
        ({"HALLUCINATED": 0, "FAITHFUL": 1}, None, "id2label"),
        ({"HALLUCINATED": 0}, {0: "HALLUCINATED"}, "FAITHFUL"),
        (
            {"HALLUCINATED": 0, "FAITHFUL": 1},
            {0: "FAITHFUL", 1: "HALLUCINATED"},
            "disagree",
        ),
        (
            {"HALLUCINATED": 0, "FAITHFUL": 0},
            {0: "FAITHFUL", 1: "HALLUCINATED"},
            "multiple labels",
        ),
    ],
)
def test_rejects_missing_ambiguous_or_inconsistent_mappings(
    label2id: object, id2label: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_faithful_logit_index(config(label2id, id2label), logits_dimension=2)


def test_rejects_faithful_index_outside_logits() -> None:
    with pytest.raises(ValueError, match="outside logits dimension"):
        resolve_faithful_logit_index(
            config({"HALLUCINATED": 0, "FAITHFUL": 2}, {0: "HALLUCINATED", 2: "FAITHFUL"}),
            logits_dimension=2,
        )


class FakeTokenizer:
    vocab_size = 32_000

    def __len__(self) -> int:
        return 32_000


def test_vocabulary_compatibility() -> None:
    assert validate_tokenizer_model_vocabulary(
        FakeTokenizer(), SimpleNamespace(vocab_size=32_000)
    ) == 32_000


def test_vocabulary_mismatch_fails() -> None:
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        validate_tokenizer_model_vocabulary(
            FakeTokenizer(), SimpleNamespace(vocab_size=31_999)
        )
