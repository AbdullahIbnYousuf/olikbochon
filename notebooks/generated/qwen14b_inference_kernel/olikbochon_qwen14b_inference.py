"""Offline Qwen3-14B-AWQ inference for Olikbochon.

The script never prints or displays competition examples. It produces only
metadata, probabilities, route flags, and a schema-validated submission.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REF = "qwen-lm/qwen-3/transformers/14b-awq/1"
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


def discover_model() -> Path:
    matches: list[Path] = []
    for config_path in Path("/kaggle/input").rglob("config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_type = str(config.get("model_type", "")).casefold()
        architectures = " ".join(map(str, config.get("architectures", []))).casefold()
        if "qwen3" in model_type or "qwen3" in architectures:
            matches.append(config_path.parent)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one attached Qwen3 model directory, found {len(unique)}")
    return unique[0]


def discover_inputs() -> tuple[Path, Path]:
    test_matches: list[Path] = []
    sample_matches: list[Path] = []
    for raw_path in glob.glob("/kaggle/input/**/*.csv", recursive=True):
        path = Path(raw_path)
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        columns = list(header.columns)
        if columns == ["id", "context", "prompt_bn", "response_bn"]:
            test_matches.append(path)
        elif columns == ["id", "label"]:
            sample_matches.append(path)
    if len(test_matches) != 1 or len(sample_matches) != 1:
        raise RuntimeError(
            f"Expected one test and sample submission file; found {len(test_matches)} and {len(sample_matches)}"
        )
    return test_matches[0], sample_matches[0]


def truncate_chars(value: object, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return text[:half] + "\n[...TRUNCATED...]\n" + text[-half:]


def user_prompt(row: pd.Series) -> str:
    context = truncate_chars(row["context"], 3600) if has_context(row["context"]) else "[NO_CONTEXT]"
    return (
        "Classify this item. Return only 0 or 1.\n"
        f"Context: {context}\n"
        f"Prompt: {truncate_chars(row['prompt_bn'], 1100)}\n"
        f"Response: {truncate_chars(row['response_bn'], 2800)}\n"
        "Label:"
    )


@torch.inference_mode()
def score(frame: pd.DataFrame, tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> np.ndarray:
    id0 = tokenizer.encode("0", add_special_tokens=False)
    id1 = tokenizer.encode("1", add_special_tokens=False)
    if len(id0) != 1 or len(id1) != 1:
        raise RuntimeError(f"Digit labels must be one token; got {len(id0)} and {len(id1)}")

    probabilities: list[float] = []
    for start in range(0, len(frame), BATCH_SIZE):
        batch = frame.iloc[start : start + BATCH_SIZE]
        rendered = []
        for _, row in batch.iterrows():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(row)},
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
        if (start // BATCH_SIZE) % 25 == 0:
            torch.cuda.empty_cache()
    return np.asarray(probabilities, dtype=np.float64)


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing CPU inference")

    model_path = discover_model()
    test_path, sample_path = discover_inputs()
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    expected_columns = ["id", "context", "prompt_bn", "response_bn"]
    if list(test.columns) != expected_columns:
        raise RuntimeError("Unexpected test schema")
    if list(sample.columns) != ["id", "label"]:
        raise RuntimeError("Unexpected sample submission schema")
    if len(test) != len(sample) or test["id"].duplicated().any() or sample["id"].duplicated().any():
        raise RuntimeError("Invalid row count or duplicate IDs")
    if not test["id"].reset_index(drop=True).equals(sample["id"].reset_index(drop=True)):
        raise RuntimeError("Test and sample IDs are not aligned")

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

    raw_probability = score(test, tokenizer, model)
    context_present = test["context"].map(has_context).to_numpy(dtype=bool)
    exact_mask = test.apply(exact_supported, axis=1).to_numpy(dtype=bool)
    probability = np.where(exact_mask, 0.995, raw_probability)
    label = (probability >= 0.5).astype(np.int64)

    probability_path = OUTPUT_ROOT / "qwen14b_test_probabilities.csv"
    pd.DataFrame(
        {
            "id": test["id"],
            "probability_label1_qwen_raw": raw_probability,
            "probability_label1_qwen_exact_override": probability,
            "context_present": context_present.astype(np.int64),
            "exact_supported": exact_mask.astype(np.int64),
        }
    ).to_csv(probability_path, index=False)

    submission_path = OUTPUT_ROOT / "submission.csv"
    submission = pd.DataFrame({"id": test["id"], "label": label})
    submission.to_csv(submission_path, index=False)
    reloaded = pd.read_csv(submission_path)
    if list(reloaded.columns) != ["id", "label"] or len(reloaded) != len(test):
        raise RuntimeError("Submission reload validation failed")
    if not set(reloaded["label"].unique()).issubset({0, 1}):
        raise RuntimeError("Submission labels are not binary")

    model_bytes = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
    summary = {
        "status": "complete",
        "model_ref": MODEL_REF,
        "model_bytes": model_bytes,
        "test_rows": len(test),
        "test_columns": expected_columns,
        "test_sha256": sha256(test_path),
        "sample_submission_sha256": sha256(sample_path),
        "example_text_printed": False,
        "external_api_calls": 0,
        "internet_required": False,
        "context_present_rows": int(context_present.sum()),
        "exact_supported_rows": int(exact_mask.sum()),
        "prediction_label1_share": float(label.mean()),
        "runtime_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "probabilities_sha256": sha256(probability_path),
        "submission_sha256": sha256(submission_path),
    }
    (OUTPUT_ROOT / "qwen14b_inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    safe_keys = [
        "status", "model_ref", "model_bytes", "test_rows", "test_columns",
        "test_sha256", "example_text_printed", "external_api_calls", "internet_required",
        "context_present_rows", "exact_supported_rows", "prediction_label1_share",
        "runtime_seconds", "gpu", "torch_version", "transformers_version",
        "probabilities_sha256", "submission_sha256",
    ]
    print(json.dumps({key: summary[key] for key in safe_keys}, indent=2))


if __name__ == "__main__":
    main()
