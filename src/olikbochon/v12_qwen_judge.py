"""Offline Qwen3 judge with frozen AA/AB decoding and compact JSON parsing."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .v12_demo_retrieval import RetrievedSet


SYSTEM = """You are a strict Bengali short-answer verification judge.
First independently determine what an acceptable answer to the prompt would be.
Then judge whether the proposed response is acceptable.
Do not judge only from writing style or lexical similarity.
For factual questions, verify entities, relations, numbers, dates, rankings, and negation.
For creative or subjective prompts, do not require a unique factual answer.
Return only one compact JSON object:
{\"label\":0_or_1,\"confidence\":number_between_0_and_1}"""


@dataclass(frozen=True)
class Target:
    official_index: int | None
    identifier: object
    prompt: str
    response: str
    retrieved: RetrievedSet


@dataclass(frozen=True)
class JudgeOutput:
    label: int
    confidence: float
    exact_template_override: bool
    invalid_outputs: int


def _user_prompt(target: Target) -> str:
    lines = [
        "Operational labels:",
        "label 1 = acceptable answer; concise, generic, uncertain, subjective, or creative is allowed when appropriate.",
        "label 0 = factual/entity/number/date error, unsupported specificity, irrelevance, contradiction, malformed answer, or instruction failure.",
        "Context is absent.",
        "Labeled demonstrations:",
    ]
    for number, demo in enumerate(target.retrieved.demonstrations, start=1):
        lines.extend(
            [
                f"Demo {number} prompt: {demo.prompt}",
                f"Demo {number} proposed response: {demo.response}",
                f"Demo {number} label: {demo.label}",
            ]
        )
    lines.extend(
        [
            f"Target prompt: {target.prompt}",
            f"Target proposed response: {target.response}",
            "Return only the compact JSON object.",
        ]
    )
    return "\n".join(lines)


def parse_compact_json(text: str) -> tuple[int, float] | None:
    """Parse exact or repaired final JSON without persisting generated reasoning."""
    candidates = re.findall(r"\{[^{}]{1,200}\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        repaired = candidate.replace("'", '"')
        repaired = re.sub(r"\b(label|confidence)\b\s*:", r'"\1":', repaired)
        try:
            value = json.loads(repaired)
            label = int(value["label"])
            confidence = float(value["confidence"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if label in (0, 1) and 0.0 <= confidence <= 1.0:
            return label, confidence
    return None


class OfflineQwenJudge:
    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.peak_vram = 0

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def _generate(self, targets: Sequence[Target], *, thinking: bool, seed: int | None) -> list[str]:
        messages = [
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _user_prompt(t)}]
            for t in targets
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
            )
            for message in messages
        ]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": 256 if thinking else 32,
            "do_sample": thinking,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if thinking:
            kwargs.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        self.peak_vram = max(self.peak_vram, int(torch.cuda.max_memory_allocated()))
        prompt_width = int(encoded["input_ids"].shape[1])
        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in output
        ]

    def judge_aa(self, targets: Sequence[Target], *, start_batch_size: int = 4) -> tuple[list[JudgeOutput], int]:
        outputs: list[JudgeOutput] = []
        batch_size = start_batch_size
        position = 0
        while position < len(targets):
            batch = targets[position : position + batch_size]
            overrides = [target.retrieved.exact_override for target in batch]
            unresolved = [target for target, override in zip(batch, overrides, strict=True) if override is None]
            try:
                generated = self._generate(unresolved, thinking=False, seed=None) if unresolved else []
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                continue
            iterator = iter(generated)
            for override in overrides:
                if override is not None:
                    outputs.append(JudgeOutput(int(override), 1.0, True, 0))
                    continue
                parsed = parse_compact_json(next(iterator))
                outputs.append(
                    JudgeOutput(0, 0.5, False, 1)
                    if parsed is None
                    else JudgeOutput(parsed[0], parsed[1], False, 0)
                )
            position += len(batch)
        return outputs, batch_size

    def judge_ab(self, targets: Sequence[Target]) -> list[JudgeOutput]:
        outputs: list[JudgeOutput] = []
        for target in targets:
            if target.retrieved.exact_override is not None:
                outputs.append(JudgeOutput(int(target.retrieved.exact_override), 1.0, True, 0))
                continue
            valid: list[tuple[int, float]] = []
            invalid = 0
            for seed in (17, 29, 43):
                parsed = parse_compact_json(self._generate([target], thinking=True, seed=seed)[0])
                if parsed is None:
                    invalid += 1
                else:
                    valid.append(parsed)
            if not valid:
                outputs.append(JudgeOutput(0, 0.5, False, invalid))
                continue
            counts = np.bincount([value[0] for value in valid], minlength=2)
            majority = int(np.argmax(counts))
            confidence = float(np.mean([value[1] for value in valid if value[0] == majority]))
            outputs.append(JudgeOutput(majority, confidence, False, invalid))
        return outputs
