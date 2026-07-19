"""Frozen offline V7 inference entry point (promotion-gate fallback: V4-A)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from olikbochon.data_loading import OFFICIAL_SAMPLE_SHA256, load_labeled_json, sha256_file
from olikbochon.v4_wikipedia import (
    ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
    V4Files,
    WikipediaRetriever,
    _wiki_chunks,
    fit_final_model,
    load_wikipedia_corpus,
    predict_probabilities,
)


TEST_COLUMNS = ("id", "context", "prompt_bn", "response_bn")
CLASSIFICATION_THRESHOLD = 0.5


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_raw_test(path: Path) -> pd.DataFrame:
    """Validate the discovered schema without assuming count, IDs, or row order."""
    if path.suffix.casefold() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.casefold() == ".json":
        frame = pd.read_json(path, orient="records")
    else:
        raise ValueError("Competition input must be CSV or records-oriented JSON")
    if set(frame.columns) != set(TEST_COLUMNS):
        raise ValueError(f"Competition input must contain exactly {TEST_COLUMNS}")
    frame = frame.loc[:, TEST_COLUMNS].reset_index(drop=True)
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Competition input IDs must be nonempty, present, and unique")
    if frame["prompt_bn"].isna().any() or frame["response_bn"].isna().any():
        raise ValueError("Competition prompts and responses must not be missing")
    return frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(f"Inference output directory already exists: {args.output_dir}")
    train = load_labeled_json(
        args.train_data, expected_sha256=OFFICIAL_SAMPLE_SHA256
    ).reset_index(drop=True)
    chunks = _wiki_chunks(args.wikipedia_root)
    if not chunks:
        raise FileNotFoundError("A valid offline WikiExtractor AA/AB/AC/AD root is required")
    placeholder = args.train_data
    corpus = load_wikipedia_corpus(
        V4Files(
            args.train_data,
            placeholder,
            placeholder,
            args.train_data.parent,
            args.wikipedia_root,
            chunks,
            0,
        )
    )
    retriever = WikipediaRetriever(corpus)
    train_retrieval = retriever.retrieve(train["prompt_bn"])
    scaler, classifier = fit_final_model(
        train,
        train_retrieval,
        retrieval_threshold=ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
    )

    # Test rows become accessible only after every train/validation-derived choice is frozen.
    test = load_raw_test(args.test_data)
    test_retrieval = retriever.retrieve(test["prompt_bn"])
    probabilities = predict_probabilities(
        test,
        test_retrieval,
        scaler,
        classifier,
        retrieval_threshold=ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
    )
    labels = (probabilities >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    submission = pd.DataFrame({"id": test["id"], "label": labels})
    if tuple(submission.columns) != ("id", "label") or len(submission) != len(test):
        raise RuntimeError("Submission schema or row count is invalid")
    if submission["id"].tolist() != test["id"].tolist() or submission["id"].duplicated().any():
        raise RuntimeError("Submission does not preserve unique input IDs and order")
    if not submission["label"].isin([0, 1]).all():
        raise RuntimeError("Submission labels must be binary")

    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "v4a_final_model.joblib"
    joblib.dump(
        {
            "scaler": scaler,
            "classifier": classifier,
            "retrieval_threshold": ORIGINAL_DEPLOYED_RETRIEVAL_THRESHOLD,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "probability_orientation": "probability_label_1_is_faithful",
        },
        model_path,
        compress=3,
    )
    submission_path = args.output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    probabilities_path = args.output_dir / "test_probabilities.csv"
    pd.DataFrame(
        {
            "id": test["id"],
            "probability_label_0": 1.0 - probabilities,
            "probability_label_1": probabilities,
        }
    ).to_csv(probabilities_path, index=False)
    counts = np.bincount(labels, minlength=2)
    summary = {
        "status": "complete_not_submitted",
        "selected_configuration": "v4a_hard_stop_fallback",
        "test_rows": len(test),
        "submission_columns": ["id", "label"],
        "prediction_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "runtime_seconds": perf_counter() - started,
        "external_api_calls": 0,
        "internet_required": False,
        "resource": {
            "name": args.resource_name,
            "version": args.resource_version,
            "license": args.resource_license,
            "source_sha1": args.resource_sha1,
            "chunk_count": len(chunks),
            "aggregate_bytes": sum(path.stat().st_size for path in chunks),
            "usable_article_count": len(corpus.articles),
        },
        "artifacts": {
            "submission": submission_path.name,
            "test_probabilities": probabilities_path.name,
            "model": model_path.name,
            "model_sha256": sha256_file(model_path),
        },
    }
    _write_json(args.output_dir / "inference_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--wikipedia-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7/final"))
    parser.add_argument("--resource-name", default="Wikimedia bnwiki pages-articles")
    parser.add_argument("--resource-version", default="20260701")
    parser.add_argument("--resource-license", default="CC-BY-SA-4.0 and GFDL-1.3-or-later")
    parser.add_argument("--resource-sha1", default="fdcf44a8ec36fc8a82b1b7b9f44878c4cb6dcba9")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "status": summary["status"],
                "test_rows": summary["test_rows"],
                "submission_columns": summary["submission_columns"],
                "external_api_calls": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
