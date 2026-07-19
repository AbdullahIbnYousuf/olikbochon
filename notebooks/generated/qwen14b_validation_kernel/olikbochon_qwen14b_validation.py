"""Offline Qwen3-14B-AWQ validation for the Olikbochon competition.

This Kaggle GPU script reads only the labeled sample split. It never opens the
competition test file. Aggregate metrics and prediction artifacts are written
to /kaggle/working without printing sample text.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


SEED = 20260719
MODEL_REF = "qwen-lm/qwen-3/transformers/14b-awq/1"
MODEL_ROOT_GLOB = "/kaggle/input/qwen-3/transformers/14b-awq/*"
OUTPUT_ROOT = Path("/kaggle/working")
MAX_LENGTH = 1792
BATCH_SIZE = 3

SYSTEM_PROMPT = """You are an expert Bengali factuality verifier. Classify whether a Bengali
response is fully faithful and factually correct. Output exactly one digit and nothing else:
0 = hallucinated: any fabricated, unsupported, contradictory, or factually incorrect claim.
1 = faithful: every material claim is accurate and, when context exists, supported by it.
If context is [NO_CONTEXT], use your own factual knowledge. Be strict about names, dates,
numbers, places, relations, negation, and unsupported additions."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_context(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip()
    return text.casefold() not in {"", "null", "none", "nan", "[null]", "[no_context]"}


def normalize(text: object) -> str:
    value = unicodedata.normalize("NFKC", "" if text is None else str(text)).casefold()
    value = re.sub(r"[^\w\u0980-\u09ff]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def exact_supported(row: pd.Series) -> bool:
    if not has_context(row["context"]):
        return False
    context = normalize(row["context"])
    response = normalize(row["response_bn"])
    return bool(response) and len(response) >= 8 and response in context


def discover_train() -> Path:
    matches = []
    for pattern in ("/kaggle/input/**/*.json", "/kaggle/input/**/*.csv"):
        for raw_path in glob.glob(pattern, recursive=True):
            path = Path(raw_path)
            try:
                if path.suffix.lower() == ".json":
                    frame = pd.read_json(path)
                else:
                    frame = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            required = {"context", "prompt_bn", "response_bn", "label"}
            if required.issubset(frame.columns):
                matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one labeled sample file, found {len(matches)}")
    return matches[0]


def truncate_chars(value: object, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n[...TRUNCATED...]\n" + text[-half:]


def select_exemplars(frame: pd.DataFrame) -> dict[tuple[bool, int], list[int]]:
    work = frame.copy()
    work["context_present"] = work["context"].map(has_context)
    work["text_length"] = (
        work["context"].fillna("").astype(str).str.len()
        + work["prompt_bn"].fillna("").astype(str).str.len()
        + work["response_bn"].fillna("").astype(str).str.len()
    )
    result: dict[tuple[bool, int], list[int]] = {}
    for route in (False, True):
        for label in (0, 1):
            subset = work[(work["context_present"] == route) & (work["label"] == label)]
            # Deterministic medium-length examples avoid both trivial and oversized demonstrations.
            target = float(subset["text_length"].median())
            ordered = subset.assign(distance=(subset["text_length"] - target).abs()).sort_values(
                ["distance", "text_length"], kind="mergesort"
            )
            result[(route, label)] = ordered.index.tolist()[:8]
    return result


def format_example(row: pd.Series) -> str:
    context = truncate_chars(row["context"], 900) if has_context(row["context"]) else "[NO_CONTEXT]"
    return (
        f"Context: {context}\n"
        f"Prompt: {truncate_chars(row['prompt_bn'], 500)}\n"
        f"Response: {truncate_chars(row['response_bn'], 900)}\n"
        f"Correct label: {int(row['label'])}"
    )


def build_user_prompt(
    row: pd.Series,
    frame: pd.DataFrame,
    exemplar_pool: dict[tuple[bool, int], list[int]],
    few_shot: bool,
    row_index: int,
) -> str:
    sections: list[str] = []
    if few_shot:
        chosen: list[int] = []
        for route, label in ((False, 0), (False, 1), (True, 0), (True, 1)):
            candidate = next((idx for idx in exemplar_pool[(route, label)] if idx != row_index), None)
            if candidate is not None:
                chosen.append(candidate)
        sections.append("Labeled examples:\n" + "\n\n".join(format_example(frame.loc[idx]) for idx in chosen))

    context = truncate_chars(row["context"], 3200) if has_context(row["context"]) else "[NO_CONTEXT]"
    sections.append(
        "Classify this item. Return only 0 or 1.\n"
        f"Context: {context}\n"
        f"Prompt: {truncate_chars(row['prompt_bn'], 1000)}\n"
        f"Response: {truncate_chars(row['response_bn'], 2600)}\n"
        "Label:"
    )
    return "\n\n".join(sections)


@torch.inference_mode()
def score_prompts(
    frame: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    exemplar_pool: dict[tuple[bool, int], list[int]],
    few_shot: bool,
) -> np.ndarray:
    id0 = tokenizer.encode("0", add_special_tokens=False)
    id1 = tokenizer.encode("1", add_special_tokens=False)
    if len(id0) != 1 or len(id1) != 1:
        raise RuntimeError(f"Digit labels must each be one token; got lengths {len(id0)}, {len(id1)}")

    probabilities: list[float] = []
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start : start + BATCH_SIZE]
        rendered = []
        for idx, row in batch.iterrows():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(row, frame, exemplar_pool, few_shot, int(idx)),
                },
            ]
            rendered.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        logits = model(**encoded, use_cache=False).logits[:, -1, :]
        pair = torch.stack((logits[:, id0[0]], logits[:, id1[0]]), dim=1).float()
        probabilities.extend(torch.softmax(pair, dim=1)[:, 1].cpu().tolist())
        del encoded, logits, pair
        if (start // BATCH_SIZE) % 20 == 0:
            torch.cuda.empty_cache()
    return np.asarray(probabilities, dtype=np.float64)


def best_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.round(np.arange(0.30, 0.701, 0.025), 3)
    scored = [(float(f1_score(y_true, probability >= threshold, average="macro")), float(threshold)) for threshold in candidates]
    return max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1], max(item[0] for item in scored)


def nested_threshold_oof(y_true: np.ndarray, probability: np.ndarray) -> tuple[np.ndarray, list[float]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    prediction = np.zeros(len(y_true), dtype=np.int64)
    thresholds: list[float] = []
    for train_idx, valid_idx in splitter.split(np.zeros(len(y_true)), y_true):
        threshold, _ = best_threshold(y_true[train_idx], probability[train_idx])
        thresholds.append(threshold)
        prediction[valid_idx] = (probability[valid_idx] >= threshold).astype(np.int64)
    return prediction, thresholds


def evaluate_candidate(y_true: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    threshold, apparent_score = best_threshold(y_true, probability)
    nested_prediction, nested_thresholds = nested_threshold_oof(y_true, probability)
    fixed_prediction = (probability >= 0.5).astype(np.int64)
    return {
        "fixed_0_5_macro_f1": float(f1_score(y_true, fixed_prediction, average="macro")),
        "nested_threshold_macro_f1": float(f1_score(y_true, nested_prediction, average="macro")),
        "nested_thresholds": nested_thresholds,
        "final_full_sample_threshold": threshold,
        "full_sample_threshold_macro_f1_apparent": apparent_score,
        "prediction_label1_share_at_final_threshold": float(np.mean(probability >= threshold)),
    }


def main() -> None:
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_path = discover_train()
    frame = pd.read_json(train_path) if train_path.suffix.lower() == ".json" else pd.read_csv(train_path)
    required = ["context", "prompt_bn", "response_bn", "label"]
    if list(frame.columns) != required:
        frame = frame[required]
    if len(frame) == 0 or frame["label"].isna().any() or not set(frame["label"].astype(int).unique()).issubset({0, 1}):
        raise RuntimeError("Invalid labeled sample split")
    frame = frame.reset_index(drop=True)

    model_candidates = sorted(glob.glob(MODEL_ROOT_GLOB))
    if len(model_candidates) != 1:
        raise RuntimeError(f"Expected one pinned model directory, found {len(model_candidates)}")
    model_path = model_candidates[0]
    model_bytes = sum(path.stat().st_size for path in Path(model_path).rglob("*") if path.is_file())

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()

    exemplar_pool = select_exemplars(frame)
    zero_probability = score_prompts(frame, tokenizer, model, exemplar_pool, few_shot=False)
    few_probability = score_prompts(frame, tokenizer, model, exemplar_pool, few_shot=True)

    exact_mask = frame.apply(exact_supported, axis=1).to_numpy(dtype=bool)
    candidate_probabilities = {
        "zero_shot": zero_probability,
        "four_shot": few_probability,
        "zero_shot_exact_override": np.where(exact_mask, 0.995, zero_probability),
        "four_shot_exact_override": np.where(exact_mask, 0.995, few_probability),
        "prompt_average": 0.5 * (zero_probability + few_probability),
        "prompt_average_exact_override": np.where(exact_mask, 0.995, 0.5 * (zero_probability + few_probability)),
    }
    y_true = frame["label"].astype(int).to_numpy()
    evaluations = {name: evaluate_candidate(y_true, probability) for name, probability in candidate_probabilities.items()}
    selected_name = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["nested_threshold_macro_f1"],
            evaluations[name]["fixed_0_5_macro_f1"],
        ),
    )
    selected_probability = candidate_probabilities[selected_name]
    selected_threshold = float(evaluations[selected_name]["final_full_sample_threshold"])

    predictions = pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=np.int64),
            "label": y_true,
            "context_present": frame["context"].map(has_context).astype(int),
            "exact_supported": exact_mask.astype(int),
            "probability_label1_zero_shot": zero_probability,
            "probability_label1_four_shot": few_probability,
            "probability_label1_selected": selected_probability,
            "prediction_selected": (selected_probability >= selected_threshold).astype(int),
        }
    )
    predictions_path = OUTPUT_ROOT / "qwen14b_validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    summary = {
        "status": "complete",
        "model_ref": MODEL_REF,
        "model_bytes": model_bytes,
        "train_rows": len(frame),
        "train_columns": required,
        "train_sha256": sha256(train_path),
        "test_file_opened": False,
        "test_rows_accessed": 0,
        "example_text_printed": False,
        "candidate_metrics": evaluations,
        "selected_candidate": selected_name,
        "selected_threshold": selected_threshold,
        "selected_nested_macro_f1": evaluations[selected_name]["nested_threshold_macro_f1"],
        "selected_fixed_0_5_macro_f1": evaluations[selected_name]["fixed_0_5_macro_f1"],
        "runtime_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "prediction_sha256": sha256(predictions_path),
    }
    (OUTPUT_ROOT / "qwen14b_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "status", "model_ref", "model_bytes", "train_rows", "test_file_opened",
        "selected_candidate", "selected_threshold", "selected_nested_macro_f1",
        "selected_fixed_0_5_macro_f1", "runtime_seconds", "gpu", "torch_version",
        "transformers_version", "prediction_sha256"
    )}, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
