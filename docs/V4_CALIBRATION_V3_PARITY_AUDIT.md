# V4 Calibration and Historical V3 Parity Audit

## Scope

This audit used only aggregate artifacts from the completed V3-compatible reproduction
and offline inference over its 15 selected checkpoints. It used the authenticated official
299-row labeled sample only. No training, test inference, public data, row text, or
row-level probability persistence occurred.

Subsequent route audit correction (2026-07-18): these diagnostics remain numerically
valid for the saved checkpoints and the legacy V3-compatible preprocessing that actually
ran. They do not validate schema-aware V4 preprocessing, because 167 literal `[NULL]`
contexts were encoded as present. Any route-specific interpretation is invalid. The
historical-control run is paused pending approval after this correction.

## Calibration findings

Nine of 15 fold checkpoints predicted label 1 for every validation row at threshold 0.50.
Fold probability means were tightly concentrated from `0.525713` to `0.568615`; Brier
scores were close to the uninformative `0.25` reference. This supports a calibration-shift
diagnosis, but the fold-optimal thresholds vary enough that `0.54` is not a universal
per-fold correction.

Selected epochs were `{1: 3, 2: 2, 3: 10}`. Best frozen-grid thresholds were
`{0.50: 2, 0.52: 3, 0.54: 3, 0.56: 3, 0.58: 4}`. Under the predeclared diagnostic
definition—best threshold within `0.02` of 0.54 and macro-F1 gap no greater than `0.02`—
threshold 0.54 was near-optimal in only 3 of 15 folds. Mean Brier score was
`0.244618660813009` with population standard deviation `0.0031485317330620186`, minimum
`0.2397481840913001`, and maximum `0.25028956046168277`.

### Probability distributions and prevalence

`True1`, `Pred1@50`, and `Pred1@54` are label-1 prevalence values.

| Seed/fold | Epoch | Min | Max | Mean | Median | Std | True1 | Pred1@50 | Pred1@54 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17/1 | 3 | 0.502079 | 0.618550 | 0.549280 | 0.524745 | 0.041330 | 0.540984 | 1.000000 | 0.409836 |
| 17/2 | 3 | 0.518759 | 0.608303 | 0.558126 | 0.536876 | 0.032196 | 0.542373 | 1.000000 | 0.474576 |
| 17/3 | 3 | 0.507720 | 0.597725 | 0.553990 | 0.544355 | 0.024719 | 0.550000 | 1.000000 | 0.633333 |
| 17/4 | 2 | 0.505562 | 0.581437 | 0.536880 | 0.525683 | 0.025347 | 0.550000 | 1.000000 | 0.450000 |
| 17/5 | 1 | 0.508003 | 0.563033 | 0.528509 | 0.521881 | 0.017105 | 0.542373 | 1.000000 | 0.322034 |
| 29/1 | 3 | 0.519769 | 0.594198 | 0.553377 | 0.545876 | 0.020650 | 0.540984 | 1.000000 | 0.606557 |
| 29/2 | 3 | 0.481073 | 0.634770 | 0.540280 | 0.507637 | 0.050817 | 0.542373 | 0.728814 | 0.389831 |
| 29/3 | 3 | 0.508315 | 0.599954 | 0.547493 | 0.533208 | 0.028738 | 0.542373 | 1.000000 | 0.474576 |
| 29/4 | 2 | 0.469954 | 0.609553 | 0.529164 | 0.500362 | 0.048453 | 0.550000 | 0.500000 | 0.383333 |
| 29/5 | 3 | 0.473350 | 0.623004 | 0.536315 | 0.504299 | 0.053130 | 0.550000 | 0.550000 | 0.450000 |
| 43/1 | 1 | 0.506935 | 0.582669 | 0.537903 | 0.525548 | 0.025479 | 0.540984 | 1.000000 | 0.426230 |
| 43/2 | 3 | 0.544809 | 0.601242 | 0.568615 | 0.560117 | 0.017628 | 0.550000 | 1.000000 | 1.000000 |
| 43/3 | 3 | 0.470600 | 0.627324 | 0.542579 | 0.513760 | 0.054219 | 0.542373 | 0.576271 | 0.474576 |
| 43/4 | 3 | 0.485158 | 0.582001 | 0.525713 | 0.504055 | 0.037115 | 0.550000 | 0.516667 | 0.416667 |
| 43/5 | 1 | 0.489685 | 0.579133 | 0.527139 | 0.510988 | 0.028179 | 0.542373 | 0.881356 | 0.389831 |

### Probability quantiles

| Seed/fold | 0.01 | 0.05 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 0.95 | 0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17/1 | 0.502705 | 0.504149 | 0.508586 | 0.514999 | 0.524745 | 0.595963 | 0.608041 | 0.612513 | 0.618274 |
| 17/2 | 0.519804 | 0.523409 | 0.526018 | 0.528927 | 0.536876 | 0.590831 | 0.600672 | 0.604346 | 0.607983 |
| 17/3 | 0.511805 | 0.521360 | 0.525472 | 0.533940 | 0.544355 | 0.579300 | 0.586209 | 0.589657 | 0.594410 |
| 17/4 | 0.506705 | 0.508549 | 0.509363 | 0.512283 | 0.525683 | 0.562700 | 0.570126 | 0.571297 | 0.578593 |
| 17/5 | 0.508202 | 0.509663 | 0.510171 | 0.513179 | 0.521881 | 0.544714 | 0.556014 | 0.556707 | 0.561124 |
| 29/1 | 0.522617 | 0.526099 | 0.531316 | 0.536009 | 0.545876 | 0.574126 | 0.580278 | 0.582268 | 0.590977 |
| 29/2 | 0.484815 | 0.491309 | 0.493733 | 0.498408 | 0.507637 | 0.595078 | 0.614104 | 0.620774 | 0.630385 |
| 29/3 | 0.509576 | 0.510877 | 0.513815 | 0.521563 | 0.533208 | 0.574581 | 0.585015 | 0.588749 | 0.598644 |
| 29/4 | 0.471606 | 0.480101 | 0.483126 | 0.487599 | 0.500362 | 0.583496 | 0.596430 | 0.604456 | 0.608610 |
| 29/5 | 0.474541 | 0.480385 | 0.483741 | 0.488107 | 0.504299 | 0.594301 | 0.605478 | 0.613579 | 0.622344 |
| 43/1 | 0.507177 | 0.509421 | 0.510638 | 0.515715 | 0.525548 | 0.563558 | 0.574186 | 0.576065 | 0.582473 |
| 43/2 | 0.545287 | 0.547682 | 0.550360 | 0.554078 | 0.560117 | 0.586804 | 0.593795 | 0.596778 | 0.599583 |
| 43/3 | 0.471297 | 0.477055 | 0.483921 | 0.492522 | 0.513760 | 0.599544 | 0.609576 | 0.611741 | 0.619086 |
| 43/4 | 0.485875 | 0.487212 | 0.488422 | 0.490771 | 0.504055 | 0.567645 | 0.573872 | 0.578607 | 0.580686 |
| 43/5 | 0.490606 | 0.495607 | 0.498494 | 0.505150 | 0.510988 | 0.555789 | 0.566456 | 0.571968 | 0.578097 |

### Frozen-grid and Brier diagnostics

| Seed/fold | Best threshold | Best macro F1 | 0.54 gap | 0.54 near-optimal | Brier |
|---|---:|---:|---:|---|---:|
| 17/1 | 0.54 | 0.605603 | 0.000000 | yes | 0.242261 |
| 17/2 | 0.58 | 0.592166 | 0.032970 | no | 0.244756 |
| 17/3 | 0.58 | 0.618056 | 0.014721 | no | 0.242213 |
| 17/4 | 0.54 | 0.600000 | 0.000000 | yes | 0.243144 |
| 17/5 | 0.52 | 0.472150 | 0.007415 | no | 0.250032 |
| 29/1 | 0.56 | 0.572737 | 0.075484 | no | 0.244596 |
| 29/2 | 0.50 | 0.542636 | 0.036431 | no | 0.247434 |
| 29/3 | 0.56 | 0.643658 | 0.016647 | no | 0.240303 |
| 29/4 | 0.58 | 0.607843 | 0.009629 | no | 0.239748 |
| 29/5 | 0.50 | 0.562290 | 0.028956 | no | 0.245167 |
| 43/1 | 0.54 | 0.589723 | 0.000000 | yes | 0.243699 |
| 43/2 | 0.56 | 0.631696 | 0.276858 | no | 0.242802 |
| 43/3 | 0.58 | 0.524194 | 0.032814 | no | 0.250290 |
| 43/4 | 0.52 | 0.600000 | 0.033815 | no | 0.243919 |
| 43/5 | 0.52 | 0.506205 | 0.000000 | no | 0.248918 |

Epoch-wise training/validation trajectories and selected validation metrics are recorded
numerically in `V4_EXPERIMENT_LOG.md`. Training loss generally declined, while validation
loss and macro F1 varied materially by fold. Because 10 of 15 selected checkpoints were
epoch 3, best-epoch selection alone does not explain the shift; it can affect five folds
and remains a controlled difference for the historical diagnostic.

## Historical V3 parity audit

| Component | Historical V3 | Completed V4-protocol baseline | Finding |
|---|---|---|---|
| Base initialization | Fresh authenticated `csebuetnlp/banglabert` for every Arm-A fold | Same authenticated base for every fold | Parity |
| Serialization | V3 prompt/context-present/context versus response markers | Calls `build_transformer_pair` through `V3_COMPATIBLE` | Exact parity |
| Normalization | Audited default BanglaBERT normalizer per field | Same vendored default normalizer | Exact parity |
| Pair construction | ELECTRA pair, ordinary markers | Same pair and tokenizer | Exact parity |
| Maximum length | 512 | 256 | Intentional difference |
| Truncation | `only_first`; explicit 384-token response head/tail fallback | `only_first`; no explicit response fallback | Implementation difference, but irrelevant to these 299 rows: historical fallback count 0 and maximum response length 28 tokens |
| Groups | Same exact, prompt/context-family, and near-duplicate audit | Same audit | Parity |
| Folds | One grouped five-fold assignment, shuffle seed 42 | Three grouped five-fold assignments, seeds 17/29/43 | Protocol difference |
| Epochs | Four official epochs | Three | Schedule difference |
| Learning rate | `1e-5` | `2e-5` | Schedule difference |
| Warmup | 0.10 | 0.10 | Parity |
| Weight decay | 0.01 | 0.01 | Parity |
| Batch / accumulation | 8 / 2, effective 16 | 8 / 2, effective 16 | Parity |
| Mixed precision | CUDA fp16 autocast and GradScaler | Same | Parity |
| Optimizer/scheduler | AdamW and linear warmup/decay | Same | Parity |
| Fold checkpoint | Historical official CV reports the final fourth epoch | Best validation macro F1, then loss, then earlier epoch | Checkpoint-selection difference |
| Threshold grid | 0.20–0.80 by 0.02; macro F1, label-0 F1, distance to 0.50, lower threshold | Same grid and ordering after repeated OOF mean | Selection ordering parity; OOF aggregation differs |
| Deployment guard | Tuned gain at least 0.01, threshold within 0.40–0.60, no >90% collapse | Candidate must first pass 0.50 collapse guard; grid result is secondary | Protocol difference |
| Label mapping | 0=`HALLUCINATED`, 1=`FAITHFUL` | Same | Exact parity |
| Label-1 probability | Resolve `FAITHFUL` to classifier index 1; float softmax column 1 | Same resolver and extraction | Exact parity; focused tests pass |
| Threshold prediction | Probability `>= threshold` maps to label 1 | Same shared function | Exact parity |
| Confusion matrix | Truth on rows, prediction on columns, labels `[0,1]` | Same shared metric | Exact parity |
| Random seeding | Python, NumPy, PyTorch, CUDA seed 42; deterministic cuDNN | Same controls using the fold seed | Mechanism parity; seed values differ |
| Train/eval mode | Explicit `train()`, `eval()`, inference mode | Same | Parity |

No label inversion, probability-column error, threshold-direction error, or confusion-matrix
orientation bug was found. The evidence supports calibration/schedule/length/checkpoint
effects rather than a broken probability implementation.

## Predeclared historical-config control

Status: paused. Do not execute this control until its intentionally historical sentinel
semantics and the corrected V4 comparison policy are explicitly approved.

The proposed `v3_historical_control` uses the historical V3 preprocessing implementation,
length 512, four epochs, learning rate `1e-5`, weight decay `0.01`, warmup `0.10`, batch 8,
accumulation 2, CUDA fp16, AdamW with linear warmup/decay, and final-epoch checkpoint
retention. Threshold 0.50 remains primary; only the frozen grid is secondary.

The committed historical seed-42 fold assignment is reconstructible. This control retains
seeds 17/29/43 and five grouped folds per seed because its purpose is to isolate
length/schedule/checkpoint effects against the completed run; switching folds would add a
confound. The seed-42 historical assignment is documented, not silently substituted.

Expected work is 15 fits and about 900 optimizer steps. Estimated runtime is 4–7 minutes,
peak allocated VRAM is approximately 5–7 GiB, and each transient selected checkpoint is
about 444 MB.

## Checkpoint retention policy

For the proposed control only:

1. save the selected/final fold checkpoint;
2. reload it offline and require exact validation loss, predictions, and metrics;
3. record aggregate fold metrics, checkpoint size, and SHA-256 of `model.safetensors`;
4. retain the representative seed-17/fold-1 checkpoint;
5. delete the other 14 control checkpoints only after their individual reload and digest
   records are written.

This reduces final control storage from approximately 6.2 GiB to about 0.42 GiB plus JSON
metadata. Peak incremental control storage remains roughly one checkpoint because folds are
processed sequentially. The immutable base snapshot, the completed reproduction summary,
and all existing reproduction checkpoints remain untouched pending separate approval.
