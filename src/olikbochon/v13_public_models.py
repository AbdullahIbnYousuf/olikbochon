"""Frozen public-only V13 sparse and BanglaBERT candidates."""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .metrics import classification_metrics
from .v13_public_pool import SEPARATOR, normalize
from .v4_preprocessing import official_context_is_present
from .v5_lexical import deterministic_substring_prediction


def serialize(frame: pd.DataFrame) -> list[str]:
    return [
        f"[ROUTE_{'PRESENT' if official_context_is_present(row.context) else 'ABSENT'}] "
        f"[CONTEXT] {normalize(row.context)} [PROMPT] {normalize(row.prompt_bn)} "
        f"{SEPARATOR} [RESPONSE] {normalize(row.response_bn)}"
        for row in frame.itertuples(index=False)
    ]


def _numbers(value: object) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", normalize(value))


def structural(frame: pd.DataFrame) -> np.ndarray:
    rows = []
    for row in frame.itertuples(index=False):
        prompt = normalize(row.prompt_bn)
        response = normalize(row.response_bn)
        context = normalize(row.context)
        prompt_tokens = set(prompt.split())
        response_tokens = set(response.split())
        context_tokens = set(context.split())
        p_numbers, r_numbers = _numbers(prompt), _numbers(response)
        rows.append(
            [
                float(official_context_is_present(row.context)),
                len(response),
                len(prompt_tokens & response_tokens) / max(1, len(prompt_tokens)),
                len(context_tokens & response_tokens) / max(1, len(context_tokens)),
                float(deterministic_substring_prediction(row.context, row.response_bn)) if official_context_is_present(row.context) else 0.0,
                len(r_numbers),
                float(bool(p_numbers) and p_numbers == r_numbers),
                len(re.findall(r"\b(?:19|20)\d{2}\b", response)),
                float(bool(re.search(r"https?://|www\.|\[[0-9]+\]", response))),
                sum(response.count(value) for value in ("না", "নয়", "not", "never")),
                float(any(value in response for value in ("জানি না", "সম্ভবত", "হতে পারে", "cannot", "perhaps"))),
                float(len(response) < 25),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


class PublicLinearClassifier:
    def __init__(self) -> None:
        self.character = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 6), sublinear_tf=True, min_df=2,
            max_features=150_000, norm="l2"
        )
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), sublinear_tf=True, min_df=2,
            max_features=75_000, norm="l2"
        )
        self.scaler = StandardScaler()
        self.model = LinearSVC(class_weight="balanced", C=1.0, random_state=42)

    def fit(self, frame: pd.DataFrame) -> PublicLinearClassifier:
        texts = serialize(frame)
        numeric = self.scaler.fit_transform(structural(frame))
        matrix = hstack((self.character.fit_transform(texts), self.word.fit_transform(texts), csr_matrix(numeric)), format="csr")
        self.model.fit(matrix, frame.label.to_numpy(dtype=np.int64))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        texts = serialize(frame)
        matrix = hstack((self.character.transform(texts), self.word.transform(texts), csr_matrix(self.scaler.transform(structural(frame)))), format="csr")
        return self.model.predict(matrix).astype(np.int64)


def _probabilities(model: Any, loader: DataLoader) -> np.ndarray:
    blocks = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            ids, mask = (value.cuda(non_blocking=True) for value in batch[:2])
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask).logits
            blocks.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(blocks)


def _loader(tokenizer: Any, texts: Sequence[str], *, batch_size: int, shuffle: bool, labels: np.ndarray | None = None) -> DataLoader:
    encoded = tokenizer(list(texts), padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(labels, dtype=torch.long))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


def probe_batch_size(model_path: Path) -> tuple[int, list[dict[str, Any]]]:
    records = []
    selected = None
    total = torch.cuda.get_device_properties(0).total_memory
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, num_labels=2
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for batch_size in (16, 32, 64):
        try:
            torch.cuda.reset_peak_memory_stats()
            ids = torch.ones((batch_size, 256), dtype=torch.long, device="cuda")
            mask = torch.ones_like(ids)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=mask, labels=torch.zeros(batch_size, dtype=torch.long, device="cuda")).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            peak = int(torch.cuda.max_memory_allocated())
            safe = peak / total < 0.90
            records.append({"batch_size": batch_size, "peak_vram_bytes": peak, "safe": safe})
            if safe:
                selected = batch_size
        except torch.OutOfMemoryError:
            records.append({"batch_size": batch_size, "oom": True, "safe": False})
            torch.cuda.empty_cache()
            break
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    if selected is None:
        raise MemoryError("V13 BanglaBERT cannot fit batch size 16")
    return selected, records


@dataclass(frozen=True)
class NeuralResult:
    validation_probability: np.ndarray
    official_probability: np.ndarray
    seed_records: list[dict[str, Any]]
    checkpoint_paths: list[str]
    batch_size: int
    probe: list[dict[str, Any]]
    peak_vram_bytes: int


def train_public_banglabert(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    official: pd.DataFrame,
    *,
    model_path: Path,
    checkpoint_root: Path,
) -> NeuralResult:
    batch_size, probe = probe_batch_size(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    train_texts, validation_texts, official_texts = serialize(train), serialize(validation), serialize(official)
    train_labels = train.label.to_numpy(dtype=np.int64)
    validation_labels = validation.label.to_numpy(dtype=np.int64)
    imbalance = max(np.bincount(train_labels, minlength=2)) / len(train_labels) > 0.55
    validation_blocks, official_blocks, records, checkpoints = [], [], [], []
    peak_vram = 0
    for seed in (17, 29, 43):
        started = perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, num_labels=2
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        train_loader = _loader(tokenizer, train_texts, batch_size=batch_size, shuffle=True, labels=train_labels)
        validation_loader = _loader(tokenizer, validation_texts, batch_size=batch_size, shuffle=False)
        official_loader = _loader(tokenizer, official_texts, batch_size=batch_size, shuffle=False)
        total_steps = 5 * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.10), total_steps)
        class_weights = None
        if imbalance:
            counts = np.bincount(train_labels, minlength=2)
            class_weights = torch.tensor(len(train_labels) / (2.0 * counts), dtype=torch.float32, device="cuda")
        best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
        epochs = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, 6):
            model.train()
            losses = []
            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.cuda(non_blocking=True), mask.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu()))
            probability = _probabilities(model, validation_loader)
            metrics = classification_metrics(validation_labels, (probability >= 0.5).astype(np.int64))
            epochs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics})
            if metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 1:
                    break
        if best_state is None:
            raise RuntimeError("V13 XC failed to select a public-validation checkpoint")
        model.load_state_dict(best_state)
        validation_blocks.append(_probabilities(model, validation_loader))
        official_blocks.append(_probabilities(model, official_loader))
        checkpoint = checkpoint_root / f"seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint)
        checkpoints.append(str(checkpoint))
        peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
        records.append({"seed": seed, "selected_epoch": best_epoch, "epochs": epochs, "runtime_seconds": perf_counter() - started})
        del best_state, optimizer, scheduler, model
        gc.collect()
        torch.cuda.empty_cache()
    return NeuralResult(
        np.mean(validation_blocks, axis=0), np.mean(official_blocks, axis=0), records,
        checkpoints, batch_size, probe, peak_vram,
    )


def infer_banglabert_test(
    test: pd.DataFrame, *, model_path: Path, checkpoint_paths: Sequence[str], batch_size: int
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    loader = _loader(tokenizer, serialize(test), batch_size=batch_size, shuffle=False)
    blocks = []
    for path in checkpoint_paths:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, num_labels=2
        ).cuda()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        blocks.append(_probabilities(model, loader))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return np.mean(blocks, axis=0)
