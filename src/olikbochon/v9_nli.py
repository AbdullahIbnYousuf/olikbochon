"""Authenticated, frozen multilingual NLI inference for V9."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .v7_retrieval import RetrievedPassage


MAX_LENGTH = 384
TOP_K = 5
BATCH_PROBES = (16, 32, 64)
MAXIMUM_VRAM_FRACTION = 0.85
DATALOADER_WORKERS = 2
NLI_FEATURES = (
    "maximum_entailment_probability",
    "mean_top3_entailment_probability",
    "maximum_contradiction_probability",
    "mean_top3_contradiction_probability",
    "maximum_entailment_minus_contradiction_margin",
    "retrieval_weighted_entailment_probability",
    "entailment_dominant_passage_count",
)


@dataclass(frozen=True)
class ModelIdentity:
    repository_id: str
    revision: str
    license: str
    content_manifest_sha256: str
    weight_sha256: str
    weight_bytes: int


@dataclass(frozen=True)
class NLIInferenceResult:
    features: np.ndarray
    diagnostics: dict[str, Any]
    identity: ModelIdentity


class _EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, encodings: dict[str, torch.Tensor]) -> None:
        self.encodings = encodings
        lengths = {int(value.shape[0]) for value in encodings.values()}
        if len(lengths) != 1:
            raise ValueError("Encoded NLI tensors are not aligned")
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.encodings.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_snapshot(path: Path) -> ModelIdentity:
    """Authenticate every immutable snapshot file against its local manifest."""
    root = Path(path).resolve()
    manifest_path = root / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_names = {str(record["path"]) for record in manifest["files"]}
    actual_names = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name != manifest_path.name
    }
    if actual_names != expected_names:
        raise ValueError("NLI snapshot file set differs from its authenticated manifest")
    canonical: list[str] = []
    for record in manifest["files"]:
        candidate = root / str(record["path"])
        size = candidate.stat().st_size
        digest = _sha256(candidate)
        if size != int(record["size"]) or digest != str(record["sha256"]):
            raise ValueError(f"NLI snapshot checksum mismatch: {record['path']}")
        canonical.append(f"{record['path']}\0{size}\0{digest}\n")
    logical = hashlib.sha256("".join(canonical).encode("utf-8")).hexdigest()
    if logical != manifest["content_manifest_sha256"]:
        raise ValueError("NLI snapshot manifest fingerprint mismatch")
    weight = root / "model.safetensors"
    return ModelIdentity(
        repository_id=str(manifest["repository_id"]),
        revision=str(manifest["revision"]),
        license=str(manifest["license"]),
        content_manifest_sha256=logical,
        weight_sha256=_sha256(weight),
        weight_bytes=weight.stat().st_size,
    )


def label_indices(config: Any) -> dict[str, int]:
    mapping = {str(label).strip().lower(): int(index) for label, index in config.label2id.items()}
    required = {"entailment", "neutral", "contradiction"}
    if set(mapping) != required or len(set(mapping.values())) != 3:
        raise ValueError("NLI label orientation is ambiguous")
    inverse = {int(index): str(label).strip().lower() for index, label in config.id2label.items()}
    if inverse != {index: label for label, index in mapping.items()}:
        raise ValueError("NLI config id2label and label2id mappings conflict")
    return mapping


def _synthetic_checks(tokenizer: Any, model: Any, indices: dict[str, int]) -> dict[str, Any]:
    premise = "The sky is blue."
    hypotheses = ("The sky is blue.", "The sky is not blue.", "A dog is running.")
    encoded = tokenizer(
        [premise] * len(hypotheses),
        list(hypotheses),
        max_length=MAX_LENGTH,
        truncation="only_first",
        padding=True,
        return_tensors="pt",
    )
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().numpy()
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Synthetic NLI probabilities are invalid")
    entailment = indices["entailment"]
    contradiction = indices["contradiction"]
    if not (
        probabilities[0, entailment] > probabilities[0, contradiction]
        and probabilities[1, contradiction] > probabilities[1, entailment]
    ):
        raise ValueError("Synthetic behavior conflicts with authenticated NLI labels")
    return {
        "pair_count": len(hypotheses),
        "probability_sums_valid": True,
        "identical_pair_entailment_dominant": True,
        "negated_pair_contradiction_dominant": True,
        "official_or_corpus_text_used": False,
    }


def _probe_batch_sizes(tokenizer: Any, model: Any) -> tuple[int, list[dict[str, Any]]]:
    device = model.device
    total_vram = torch.cuda.get_device_properties(device).total_memory
    premise = "The sky is blue. " * 160
    hypothesis = "The sky is blue."
    probes: list[dict[str, Any]] = []
    eligible: list[int] = []
    for batch_size in BATCH_PROBES:
        encoded = tokenizer(
            [premise] * batch_size,
            [hypothesis] * batch_size,
            max_length=MAX_LENGTH,
            truncation="only_first",
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        failed = False
        try:
            with torch.inference_mode():
                model(**encoded)
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        except torch.cuda.OutOfMemoryError:
            failed = True
            peak = int(total_vram)
            torch.cuda.empty_cache()
        fraction = float(peak / total_vram)
        accepted = not failed and fraction < MAXIMUM_VRAM_FRACTION
        if accepted:
            eligible.append(batch_size)
        probes.append(
            {
                "batch_size": batch_size,
                "synthetic_only": True,
                "oom": failed,
                "peak_allocated_vram_bytes": peak,
                "peak_vram_fraction": fraction,
                "below_85_percent": accepted,
            }
        )
        del encoded
    if not eligible:
        raise MemoryError("No synthetic NLI batch size remained below 85% VRAM")
    return max(eligible), probes


def _aggregate_features(
    probabilities: np.ndarray,
    retrieval_scores: np.ndarray,
    indices: dict[str, int],
) -> np.ndarray:
    if probabilities.shape[:2] != retrieval_scores.shape or probabilities.shape[1] != TOP_K:
        raise ValueError("NLI probabilities and retrieval scores are not aligned top-5 rows")
    entailment = probabilities[:, :, indices["entailment"]]
    contradiction = probabilities[:, :, indices["contradiction"]]
    margins = entailment - contradiction
    weights = retrieval_scores.sum(axis=1)
    weighted = np.divide(
        (entailment * retrieval_scores).sum(axis=1),
        weights,
        out=entailment.mean(axis=1).copy(),
        where=weights > 0,
    )
    matrix = np.column_stack(
        (
            entailment.max(axis=1),
            np.sort(entailment, axis=1)[:, -3:].mean(axis=1),
            contradiction.max(axis=1),
            np.sort(contradiction, axis=1)[:, -3:].mean(axis=1),
            margins.max(axis=1),
            weighted,
            (entailment > contradiction).sum(axis=1),
        )
    ).astype(np.float64)
    if not np.isfinite(matrix).all() or matrix.shape[1] != len(NLI_FEATURES):
        raise ValueError("Frozen NLI aggregate matrix is invalid")
    return matrix


def run_frozen_nli_inference(
    model_path: Path,
    evidence: Sequence[Sequence[RetrievedPassage]],
    responses: Sequence[str],
    *,
    use_fast: bool,
) -> NLIInferenceResult:
    """Run one authenticated frozen model on exactly five passages per null row."""
    started = perf_counter()
    identity = authenticate_snapshot(model_path)
    if identity.license.lower() != "mit":
        raise ValueError("Frozen NLI snapshot does not declare the approved MIT license")
    if len(evidence) != len(responses) or any(len(row) != TOP_K for row in evidence):
        raise ValueError("Frozen NLI inference requires exactly five passages per row")
    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    indices = label_indices(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=use_fast,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    sanity = _synthetic_checks(tokenizer, model, indices)
    batch_size, probes = _probe_batch_sizes(tokenizer, model)
    premises = [passage.snippet for row in evidence for passage in row]
    hypotheses = [str(response) for response in responses for _passage in range(TOP_K)]
    encoded = tokenizer(
        premises,
        hypotheses,
        max_length=MAX_LENGTH,
        truncation="only_first",
        padding="max_length",
        return_tensors="pt",
    )
    dataset = _EncodedDataset(dict(encoded))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(model.device)
    inference_started = perf_counter()
    blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {name: value.to("cuda", non_blocking=True) for name, value in batch.items()}
            blocks.append(torch.softmax(model(**inputs).logits.float(), dim=-1).cpu().numpy())
    torch.cuda.synchronize()
    inference_seconds = perf_counter() - inference_started
    peak_vram = int(torch.cuda.max_memory_allocated(model.device))
    probabilities = np.concatenate(blocks).reshape(len(responses), TOP_K, -1)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=2), 1.0, atol=1e-5
    ):
        raise ValueError("Official-sample NLI probabilities are invalid")
    scores = np.asarray(
        [[float(passage.score) for passage in row] for row in evidence], dtype=np.float64
    )
    features = _aggregate_features(probabilities, scores, indices)
    entailment = probabilities[:, :, indices["entailment"]].ravel()
    contradiction = probabilities[:, :, indices["contradiction"]].ravel()
    neutral = probabilities[:, :, indices["neutral"]].ravel()
    flat_scores = scores.ravel()
    correlation = float(np.corrcoef(flat_scores, entailment)[0, 1])
    if not np.isfinite(correlation):
        correlation = 0.0
    diagnostics = {
        "repository_id": identity.repository_id,
        "revision": identity.revision,
        "architecture": config.architectures,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "config_max_position_embeddings": int(config.max_position_embeddings),
        "label_indices": indices,
        "synthetic_sanity": sanity,
        "batch_probes": probes,
        "selected_batch_size": batch_size,
        "dataloader_workers": DATALOADER_WORKERS,
        "pin_memory": True,
        "pair_count": int(probabilities.shape[0] * probabilities.shape[1]),
        "maximum_length": MAX_LENGTH,
        "truncation": "only_first_preserve_hypothesis",
        "entailment_probability": _summary(entailment),
        "contradiction_probability": _summary(contradiction),
        "neutral_probability": _summary(neutral),
        "entailment_minus_contradiction_margin": _summary(entailment - contradiction),
        "rows_with_entailment_dominant_passage_percentage": float(
            100.0
            * (
                probabilities[:, :, indices["entailment"]]
                > probabilities[:, :, indices["contradiction"]]
            ).any(axis=1).mean()
        ),
        "retrieval_entailment_pearson_correlation": correlation,
        "inference_seconds": inference_seconds,
        "total_model_stage_seconds": perf_counter() - started,
        "peak_allocated_vram_bytes": peak_vram,
        "gpu_name": torch.cuda.get_device_name(0),
        "total_gpu_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "network_access_used": False,
        "raw_text_persisted": False,
    }
    del loader, dataset, encoded, model, tokenizer, probabilities, blocks
    gc.collect()
    torch.cuda.empty_cache()
    return NLIInferenceResult(features, diagnostics, identity)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
