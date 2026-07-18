from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from olikbochon.champion_submission import (
    ABSENT_ROUTE,
    CHAMPION_CANDIDATE,
    CHAMPION_THRESHOLD,
    EXPECTED_ABSENT_TRAINING_ROWS,
    EXPECTED_PRESENT_TRAINING_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAINING_ROWS,
    PRESENT_ROUTE,
    PROBABILITY_COLUMNS,
    fit_frozen_champion,
    frozen_champion_config,
    predict_frozen_champion,
)
from olikbochon.champion_submission_runner import (
    MODE,
    OUTPUT_NAME,
    authenticate_test_file,
    build_cli_parser,
    validate_arguments,
)
from olikbochon.v5_features import CONTEXT_ABSENT_FEATURES
from olikbochon.v5_sparse import CANDIDATES, CHAR_SPEC, LOGISTIC_CONFIG, WORD_SPEC


def synthetic_training_frame() -> pd.DataFrame:
    absent = EXPECTED_ABSENT_TRAINING_ROWS
    present = EXPECTED_PRESENT_TRAINING_ROWS
    return pd.DataFrame(
        {
            "context": ["[NULL]"] * absent
            + [f"known evidence answer {index % 7}" for index in range(present)],
            "prompt_bn": [
                f"shared official question topic {index % 11}"
                for index in range(EXPECTED_TRAINING_ROWS)
            ],
            "response_bn": [
                f"shared null answer {'faithful' if index % 2 else 'hallucinated'} {index % 5}"
                for index in range(absent)
            ]
            + [f"answer {index % 7}" for index in range(present)],
            "label": [index % 2 for index in range(EXPECTED_TRAINING_ROWS)],
        }
    )


def synthetic_test_frame() -> pd.DataFrame:
    present = 100
    absent = EXPECTED_TEST_ROWS - present
    return pd.DataFrame(
        {
            "id": np.arange(50_000, 50_000 + EXPECTED_TEST_ROWS),
            "context": [
                (
                    f"synthetic context answer {index % 9}"
                    if index % 2 == 0
                    else "synthetic unrelated evidence"
                )
                for index in range(present)
            ]
            + ["[NULL]"] * absent,
            "prompt_bn": [
                f"secret_prompt_payload topic {index % 11}"
                for index in range(EXPECTED_TEST_ROWS)
            ],
            "response_bn": [f"answer {index % 9}" for index in range(present)]
            + [
                f"secret_response_payload null answer {index % 7}"
                for index in range(absent)
            ],
        }
    )


@pytest.fixture(scope="module")
def fitted_and_outputs():
    fitted = fit_frozen_champion(synthetic_training_frame())
    test = synthetic_test_frame()
    return fitted, test, predict_frozen_champion(fitted, test)


def test_frozen_candidate_i_configuration_is_exact() -> None:
    config = frozen_champion_config()
    assert CHAMPION_CANDIDATE == CANDIDATES[4]
    assert CHAMPION_THRESHOLD == 0.50
    assert config["character_vectorizer"] == CHAR_SPEC.__dict__
    assert config["word_vectorizer"] == WORD_SPEC.__dict__
    assert config["numeric_features"] == list(CONTEXT_ABSENT_FEATURES)
    assert len(config["numeric_features"]) == 21
    assert config["classifier"] == LOGISTIC_CONFIG


def test_full_training_fits_exactly_169_absent_rows(fitted_and_outputs) -> None:
    fitted, _test, _outputs = fitted_and_outputs
    audit = fitted.training_audit
    assert audit["official_training_rows"] == 299
    assert audit["context_present_training_rows"] == 130
    assert audit["context_absent_training_rows"] == 169
    assert audit["candidate_fit"]["training_row_count"] == 169
    assert audit["test_rows_used_for_fit"] == 0
    assert audit["classifier_classes"] == [0, 1]


def test_test_rows_never_change_fitted_vocabulary_or_rare_map(fitted_and_outputs) -> None:
    fitted, test, _outputs = fitted_and_outputs
    vocabularies = {
        name: dict(vectorizer.vocabulary_)
        for name, vectorizer in fitted.model.vectorizers.items()
    }
    assert fitted.model.numeric_extractor is not None
    rare_map = dict(fitted.model.numeric_extractor.response_token_frequency)
    predict_frozen_champion(fitted, test)
    assert vocabularies == {
        name: vectorizer.vocabulary_
        for name, vectorizer in fitted.model.vectorizers.items()
    }
    assert rare_map == fitted.model.numeric_extractor.response_token_frequency
    assert "secret_response_payload" not in rare_map


def test_present_route_uses_only_frozen_substring_rule(fitted_and_outputs) -> None:
    _fitted, _test, outputs = fitted_and_outputs
    present = outputs.probabilities["route"] == PRESENT_ROUTE
    absent = outputs.probabilities["route"] == ABSENT_ROUTE
    assert int(present.sum()) == 100
    assert outputs.probabilities.loc[present, "label_1_probability"].isna().all()
    assert outputs.probabilities.loc[present, "deterministic_rule_score"].notna().all()
    assert outputs.probabilities.loc[absent, "deterministic_rule_score"].isna().all()
    assert outputs.probabilities.loc[absent, "label_1_probability"].notna().all()
    assert outputs.probabilities.loc[present, "predicted_label"].tolist() == [
        int(index % 2 == 0) for index in range(100)
    ]


def test_submission_and_probability_schema_preserve_ids_and_order(
    fitted_and_outputs,
) -> None:
    _fitted, test, outputs = fitted_and_outputs
    assert list(outputs.submission.columns) == ["id", "label"]
    assert len(outputs.submission) == EXPECTED_TEST_ROWS
    assert outputs.submission["id"].tolist() == test["id"].tolist()
    assert outputs.submission["id"].is_unique
    assert outputs.submission["label"].isin([0, 1]).all()
    assert tuple(outputs.probabilities.columns) == PROBABILITY_COLUMNS
    assert outputs.probabilities["id"].tolist() == test["id"].tolist()
    assert outputs.probabilities["predicted_label"].tolist() == outputs.submission[
        "label"
    ].tolist()


def test_probability_artifact_contains_no_raw_synthetic_text(fitted_and_outputs) -> None:
    _fitted, _test, outputs = fitted_and_outputs
    serialized = outputs.probabilities.to_csv(index=False)
    assert "secret_prompt_payload" not in serialized
    assert "secret_response_payload" not in serialized
    assert set(outputs.probabilities["route"]) == {PRESENT_ROUTE, ABSENT_ROUTE}


def test_repeated_inference_is_exactly_deterministic(fitted_and_outputs) -> None:
    fitted, test, first = fitted_and_outputs
    second = predict_frozen_champion(fitted, test)
    pd.testing.assert_frame_equal(first.submission, second.submission)
    pd.testing.assert_frame_equal(first.probabilities, second.probabilities)
    assert first.inference_audit == second.inference_audit


def test_locked_cli_rejects_overrides_and_arbitrary_paths(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/submissions/\n", encoding="utf-8")
    train = (tmp_path / "data" / "competition" / "dataset samples.json").resolve()
    test = (tmp_path / "data" / "competition" / "test set.csv").resolve()
    output = (tmp_path / "artifacts" / "submissions" / OUTPUT_NAME).resolve()
    args = build_cli_parser().parse_args(
        [
            "--mode",
            MODE,
            "--train-path",
            str(train),
            "--test-path",
            str(test),
            "--output-dir",
            str(output),
        ]
    )
    assert validate_arguments(args, tmp_path) == (train, test, output)
    args.test_path = (tmp_path / "data" / "public-20k" / "test set.csv").resolve()
    with pytest.raises(ValueError, match="official competition test"):
        validate_arguments(args, tmp_path)
    args.test_path = test
    args.output_dir = (tmp_path / "artifacts" / "submissions" / "alternate").resolve()
    with pytest.raises(ValueError, match=OUTPUT_NAME):
        validate_arguments(args, tmp_path)
    with pytest.raises(SystemExit):
        build_cli_parser().parse_args(["--candidate", "anything"])


def test_output_must_be_ignored_and_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("artifacts/submissions/\n", encoding="utf-8")
    train = (tmp_path / "data" / "competition" / "dataset samples.json").resolve()
    test = (tmp_path / "data" / "competition" / "test set.csv").resolve()
    output = (tmp_path / "artifacts" / "submissions" / OUTPUT_NAME).resolve()
    args = build_cli_parser().parse_args(
        ["--mode", MODE, "--train-path", str(train), "--test-path", str(test), "--output-dir", str(output)]
    )
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite is prohibited"):
        validate_arguments(args, tmp_path)


def test_authentication_checks_bytes_before_csv_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"synthetic authenticated bytes"
    path = tmp_path / "test set.csv"
    path.write_bytes(payload)
    monkeypatch.setattr(
        "olikbochon.champion_submission_runner.EXPECTED_TEST_BYTES", len(payload)
    )
    monkeypatch.setattr(
        "olikbochon.champion_submission_runner.EXPECTED_TEST_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    assert authenticate_test_file(path) == {
        "filename": "test set.csv",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
