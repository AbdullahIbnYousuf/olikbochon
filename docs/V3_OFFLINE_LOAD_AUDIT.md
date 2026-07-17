# Version 3 Offline BanglaBERT Load Audit

## Decision

**Passed.** The authenticated local BanglaBERT snapshot loads as both
`ElectraForPreTraining` and `ElectraForSequenceClassification` without network
access. Every reusable ELECTRA encoder parameter loaded; only the expected
pretraining-head removal and downstream-classifier initialization occurred.

No competition data, Kaggle notebook, training loop, checkpoint save, or test
inference was used.

## Environment

| Component | Version or value |
|---|---|
| Python distribution method | Astral `uv 0.11.29` managed CPython |
| Python | `3.12.12` |
| PyTorch | `2.6.0+cpu` |
| Transformers | `4.51.3` |
| huggingface_hub | `0.36.2` |
| tokenizers | `0.21.4` |
| safetensors | `0.5.3` |
| OS | Linux `7.0.0-27-generic`, glibc 2.43 |
| CPU architecture | `x86_64` |
| CUDA available | No |

The managed interpreter was stored under `/tmp/olikbochon-python312/` and the
isolated environment under `/tmp/olikbochon-banglabert-load-venv/`. Neither
location is inside the repository. Lightning, Accelerate, Weights & Biases,
Jupyter, and GPU packages were not installed.

## Authentication and offline controls

| Control | Result |
|---|---|
| Local snapshot | `data/models/banglabert-official-9ce791f/` |
| Official revision | `9ce791f330578f50da6bc52b54205166fb5d1c8c` |
| Weight SHA-256 | `9d33f519f42705d54e65fc1601644a6f4562c3462f96943b32b4184536130f98` |
| Seven-file hash recheck | Passed before importing/loading the weight |
| `HF_HUB_OFFLINE` | `1` |
| `TRANSFORMERS_OFFLINE` | `1` |
| `local_files_only` | `True` for every tokenizer/model/config load |
| `trust_remote_code` | `False` for every tokenizer/model/config load |
| Executable model identifier | Local filesystem path only |
| Process network | Restricted sandbox plus a socket connection guard |
| Recorded socket attempts | 0 |

The rejected `reasat/banglabert` mirror was not inspected or used.

## Tokenizer

| Check | Result |
|---|---|
| Class | `ElectraTokenizerFast` |
| `len(tokenizer)` | 32,000 |
| Reported vocabulary | 32,000 |
| Padding / unknown token | `[PAD]` / `[UNK]` |
| Separator / classification / mask token | `[SEP]` / `[CLS]` / `[MASK]` |
| Pair special-token requirement | 3 |
| Token-type IDs | Supported |
| Synthetic Bengali pair | Passed; input and attention-mask shape `[2, 27]` |

The tokenizer reports the conventional very-large “no explicit tokenizer
limit” sentinel for `model_max_length`; the audited ELECTRA configuration is
the authoritative limit and declares `max_position_embeddings=512`.
Structural markers remain ordinary literal text. No tokens or special tokens
were added, and no embedding matrix was resized.

An integrated synthetic preprocessing check used the real local tokenizer with
both sequences deliberately overlength. The deterministic fallback kept 192
leading and 192 trailing Sequence B tokens, produced exactly 512 total encoded
tokens, and retained the raw missing-context flag as 0.

## Pretraining-model load

- Class: `ElectraForPreTraining`.
- Synthetic logits shape: `[2, 27]`.
- Model type: `electra`.
- Vocabulary: 32,000, matching the tokenizer.
- Hidden/embedding size: 768/768.
- Encoder: 12 layers, 12 heads, intermediate size 3,072.
- Maximum positions: 512.

Complete aggregate loading information:

| Category | Result |
|---|---|
| `missing_keys` | `[]` |
| `unexpected_keys` | `[]` |
| `mismatched_keys` | `[]` |
| `error_msgs` | `[]` |

## Sequence-classification transition

- Class: `ElectraForSequenceClassification`.
- Labels: `0=HALLUCINATED`, `1=FAITHFUL`.
- Synthetic two-example logits shape: `[2, 2]`.
- Missing `electra.*` parameters: none.
- Mismatched keys: none.
- Loading errors: none.

Expected newly initialized keys:

```text
classifier.dense.bias
classifier.dense.weight
classifier.out_proj.bias
classifier.out_proj.weight
```

Expected discarded pretraining-head keys:

```text
discriminator_predictions.dense.bias
discriminator_predictions.dense.weight
discriminator_predictions.dense_prediction.bias
discriminator_predictions.dense_prediction.weight
```

No other missing or unexpected key group occurred. The model warning that the
new classifier requires downstream training is expected; no training was
performed in this audit.

## Reusable safety helpers

`resolve_faithful_logit_index` validated both model mappings and resolved
`FAITHFUL` to logit index 1 in the audited configuration. It does not assume a
fixed probability column and fails on missing, ambiguous, inconsistent, or
out-of-range mappings. The vocabulary helper independently confirmed 32,000
for tokenizer length, tokenizer vocabulary, and model configuration.
