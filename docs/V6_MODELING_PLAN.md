# V6 official-only null-route neural plan

Status: **prepared but not executed**. This document freezes the experiment before any
V6 neural result is observed. It does not authorize training.

## Objective and data boundary

V6 tests whether low-capacity neural adaptation improves the corrected context-absent
route. It uses only the hash-authenticated official 299-row labeled sample. The neural
fit scope is the 169 context-absent rows (label 0: 89; label 1: 80). The 130
context-present rows are handled only by the frozen V5 Candidate A normalized substring
rule.

Competition-test data, public 5K/20K data, retrieval corpora, external APIs,
augmentation, predictions, and submissions are outside this experiment. The immutable
local BanglaBERT snapshot remains the only permitted model source.

## Exact null-route input

Each null-route input is serialized exactly as:

```text
[QUESTION]
<normalized prompt>

[ANSWER]
<normalized response>
```

The context field is never serialized. Literal `[NULL]`, evidence text, route flags, and
context placeholders are therefore absent. `[QUESTION]` and `[ANSWER]` are ordinary text
markers, not tokenizer special tokens. Maximum tokenized length is 192.

## Frozen validation

- seeds: 17, 29, 43;
- five folds per seed and 15 fits per candidate;
- V5 strengthened groups, joining inherited family/near-duplicate groups with normalized
  exact-row, prompt-identity, and present-context-identity groups;
- zero train/validation group overlap and both labels required in every route-specific
  train and validation split;
- fixed tokenization only; no validation-fitted preprocessing;
- class weights computed from the current null-route training fold only;
- context-present validation predictions always come from frozen Candidate A.

The preflight reconstructed 291 groups (largest group: 2), all 15 required folds, zero
overlap, and both labels in every route-specific split.

## Frozen candidates

| Candidate | Trainability policy | Exact trainable parameters | Learning rates |
|---|---|---:|---|
| J | freeze all 12 encoder layers; train a dropout-plus-linear two-class head | 1,538 | head `2e-5` |
| K | freeze embeddings and encoder layers 1–10; train layers 11–12 and the standard classifier head | 14,767,874 | head `2e-5`; top layers `5e-6` |
| L | full fine-tuning control | 110,618,882 | all parameters `1e-5` |

The authenticated standard classifier has 110,618,882 parameters. Replacing it with the
small Candidate J head yields 110,028,290 total parameters for that candidate. Candidate
L is diagnostic: its result does not retroactively tune J or K.

## Shared training and selection

- maximum six epochs;
- batch size 8, gradient accumulation 2, effective batch size 16;
- AdamW weight decay 0.01;
- linear schedule with 10% warmup;
- CUDA fp16, data-loader workers 0;
- training-fold-only balanced class weights;
- gradient norm clipping at 1.0;
- no label smoothing or focal loss;
- selected epoch ranks validation macro F1 at threshold 0.50, then lower validation loss,
  then earlier epoch.

Threshold 0.50 remains primary. Each candidate gets exactly one secondary threshold,
selected from 0.20 through 0.80 in 0.02 increments using its mean repeated-OOF null-route
probabilities. No per-fold or per-seed deployment threshold is selected. A threshold is
rejected when either predicted class exceeds 90%.

J and K pass only at null-route macro F1 at least 0.58 with no collapse. L is reported as
a diagnostic full-fine-tuning control. Comparisons are frozen against the null sparse
champion 0.574025 and routed champion 0.692051. If J or K passes, the predeclared next
direction is sparse-plus-neural hybrid evaluation; if only L reaches 0.58, it is a
null-route neural-model signal; otherwise the next direction is retrieval/NLI.

## Checkpoints and safe artifacts

All output is confined to ignored `artifacts/v6/null_neural_baseline/`. Selected
checkpoints store trainable parameter tensors only and reconstruct from the authenticated
immutable base snapshot. Every selected checkpoint must exactly reproduce probabilities,
predictions, metrics, and weighted validation loss before it is eligible for deletion.
Only seed 17/fold 1 is retained for each candidate. Fold and aggregate JSON contain numeric
metadata only; raw text, token strings, row identifiers, probabilities, and row-level
values are never persisted.

Estimated float32 delta sizes before small safetensors/JSON overhead are:

- J: 6,152 bytes;
- K: 59,071,496 bytes (56.34 MiB);
- L: 442,475,528 bytes (421.98 MiB);
- final three-checkpoint retention: about 501.6 MB (478.3 MiB);
- cumulative checkpoint bytes written across 45 fits: about 7.52 GB, while immediate
  post-verification deletion keeps peak checkpoint storage near the final 478.3 MiB.

## Resource estimate

The estimates use measured V4 runs on the same RTX 4070 SUPER as anchors: the corrected
15-fit length-256 run took 109.5 seconds and peaked at 3.48 GB allocated VRAM; the
length-512 historical control took 160.6 seconds and peaked at 6.03 GB.

For 45 V6 fits at length 192 and on only the null route, the conservative estimate is:

| Candidate | Estimated 15-fit runtime | Estimated peak allocated VRAM |
|---|---:|---:|
| J | 1–2 minutes | 1.0–1.8 GB |
| K | 1–2 minutes | 1.5–2.5 GB |
| L | 1.5–3 minutes | 2.5–3.5 GB |
| Total | 4–8 minutes | maximum 3.5 GB |

These are planning ranges, not measured V6 results. Model loading, antivirus activity,
and thermal state can affect wall time.

## Prepared command (not yet authorized)

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"

.\.venv\Scripts\python.exe -m olikbochon.v6_runner `
  --mode null-neural-baseline `
  --model-path "F:\datathon\olikbochon\data\models\banglabert-official-9ce791f" `
  --candidate-set frozen `
  --seeds 17 29 43 `
  --folds 5 `
  --max-length 192 `
  --output-dir "F:\datathon\olikbochon\artifacts\v6\null_neural_baseline"
```

Running this command requires separate explicit approval.
