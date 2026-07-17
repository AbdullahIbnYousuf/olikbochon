from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from olikbochon.metrics import classification_metrics, predictions_from_label1
from olikbochon.v3_modeling import resolve_faithful_logit_index
from olikbochon.v3_training import ID2LABEL, LABEL2ID


def test_label1_probability_is_softmax_classifier_index_one() -> None:
    config = SimpleNamespace(label2id=LABEL2ID, id2label=ID2LABEL)
    assert resolve_faithful_logit_index(config, logits_dimension=2) == 1
    logits = np.asarray([[2.0, 1.0], [-1.0, 3.0]], dtype=np.float64)
    exponentiated = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    probabilities_label1 = softmax[:, 1]
    assert probabilities_label1[0] < 0.5
    assert probabilities_label1[1] > 0.5
    assert predictions_from_label1(probabilities_label1, 0.5).tolist() == [0, 1]


def test_v3_threshold_and_confusion_orientation_are_not_inverted() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities_label1 = np.asarray([0.20, 0.80, 0.30, 0.90], dtype=np.float64)
    predictions = predictions_from_label1(probabilities_label1, 0.5)
    assert predictions.tolist() == [0, 1, 0, 1]
    metrics = classification_metrics(truth, predictions)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["f1_label0"] == 0.5
    assert metrics["f1_label1"] == 0.5
