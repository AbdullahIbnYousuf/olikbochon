from pathlib import Path

import numpy as np

from olikbochon.data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json
from olikbochon.modeling import build_model, label_probability
from olikbochon.preprocessing import build_text_series


ROOT = Path(__file__).resolve().parents[1]


def synthetic_training_data() -> tuple[list[str], np.ndarray]:
    texts = [
        f"__PROMPT__ প্রশ্ন {index} __CONTEXT__ তথ্য {index % 3} __RESPONSE__ উত্তর {index % 2}"
        for index in range(20)
    ]
    labels = np.array([index % 2 for index in range(20)], dtype=np.int64)
    return texts, labels


def test_model_predictions_are_deterministic() -> None:
    texts, labels = synthetic_training_data()
    first = build_model().fit(texts, labels)
    second = build_model().fit(texts, labels)
    assert np.array_equal(first.predict(texts), second.predict(texts))
    assert np.allclose(label_probability(first, texts, 1), label_probability(second, texts, 1))


def test_official_labeled_sample_schema_and_hash_only() -> None:
    path = ROOT / "data" / "competition" / "dataset samples.json"
    frame = load_labeled_json(path, expected_sha256=OFFICIAL_SAMPLE_SHA256)
    assert not frame.empty
    assert set(frame["label"].unique()) == {0, 1}
    assert len(build_text_series(frame)) == len(frame)
