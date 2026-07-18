# Version 4 Experiment Log

## Status

Gate 4 GPU smoke training passed. Gate 5 completed all 15 authorized fits but the
V3-compatible baseline failed the frozen threshold-0.50 class-collapse acceptance guard.
The schema-corrected rerun also completed 15 fits and failed the same primary collapse
guard; its frozen-grid threshold 0.54 result is diagnostic only.
The schema-corrected historical-schedule control completed 15 fits, failed the primary
guard, and was approximately equivalent at its selected threshold 0.52.
No submission or competition-test inference exists.

Route audit correction (2026-07-18): the authenticated sample contains 167 literal
`[NULL]` context sentinels and two actual nulls. The completed smoke and reproduction
runs inherited V3's truthy-string behavior and therefore encoded those 167 sentinels as
context-present. Their saved losses, predictions, overall metrics, reload checks, and
calibration statistics remain exact records of the pipeline that ran, and their grouped
fold indices remain unchanged. They are not valid measurements of the corrected,
schema-aware V4 preprocessing path. All recorded context-present/null subgroup metrics
and the 297/2 truncation route counts are invalid for V4 route interpretation. Corrected
V4 counts are 130 context-present and 169 context-absent. Legacy artifacts were not
deleted or reinterpreted; the isolated corrected rerun is recorded below.

## Frozen family

| Order | Configuration | Result status |
|---:|---|---|
| 1 | V3-compatible serialization, comparison encoder | Smoke passed; Gate 5 reproduction failed the 0.50 collapse guard |
| 1a | Schema-corrected V3-compatible serialization, comparison encoder | Completed; failed the 0.50 collapse guard |
| 1b | Schema-corrected preprocessing, historical V3 schedule | Completed; failed at 0.50 and approximately equivalent at selected threshold |
| 2 | Structured serialization, comparison encoder | Not run |
| 3 | Routed serialization, comparison encoder | Not run |
| 4 | Selected serialization, field-aware encoder | Not run |

Validation seeds are 17, 29, and 43 with five group-isolated folds per seed. The primary
metric is ordinary two-class macro F1. The threshold grid is 0.20–0.80 inclusive by 0.02.
The complete precommitted configuration and tie-break rules are in `V4_MODELING_PLAN.md`.

## Smoke test

Passed on 2026-07-17 at commit `584fbc0c11f6fab2bc1cc7a7c250c752fad5c73b`.

- configuration: `v3_compatible_baseline`, maximum length 256, CUDA fp16, batch size 4,
  gradient accumulation 2, two optimizer steps, threshold 0.50 only;
- split: seed 17, fold 1; 238 training rows and 61 validation rows;
- grouping: 296 total groups, 236 training groups, 60 validation groups, zero overlapping
  groups;
- bounded training: 16 rows were consumed during the two optimizer steps;
- mean observed training-batch loss: `0.6824951171875`;
- initial/final batch losses: not retained by the first CLI implementation and therefore
  unavailable without an unauthorized replay; the CLI now records both for future runs;
- validation loss: `0.6891969774590164`;
- validation metrics: macro F1 `0.35106382978723405`, label-0 F1 `0.0`, label-1 F1
  `0.7021276595744681`, accuracy `0.5409836065573771`, confusion matrix
  `[[0, 28], [0, 33]]`;
- context diagnostics: context-present macro F1 `0.3548387096774194` over 60 rows;
  null-context macro F1 `0.0` over one row;
- legacy-at-run-time token audit over 299 official rows: 297 context-present and two
  null-context; this route split is invalid for schema-aware V4 interpretation;
  no prompt, context, or response truncation was reported by the comparison encoder;
- authenticated snapshot: the approved repository-ignored BanglaBERT revision
  `9ce791f330578f50da6bc52b54205166fb5d1c8c`;
- hardware/runtime: NVIDIA GeForce RTX 4070 SUPER, PyTorch `2.11.0+cu128`, embedded CUDA
  12.8, `4.586087499999849` seconds wall clock, `2495566848` peak allocated GPU bytes;
- checkpoint: `444062308` bytes under ignored `artifacts/v4/`;
- reload: offline-safe checkpoint loading passed with zero missing/unexpected keys;
  validation loss, thresholded predictions, and aggregate metrics matched exactly;
- warnings: the diagnostic classifier predicted only label 1 on this validation fold. The
  smoke result is not evidence for model quality and was not used for selection.

## V3 reproduction under the V4 runner

Interpretation: **V3-compatible baseline evaluated under the frozen V4
repeated-grouped-validation protocol.** This is not an exact historical V3 reproduction.

The run used the authenticated official 299-row sample only: three seeds, five grouped
folds per seed, three epochs, length 256, CUDA fp16, batch size 8, accumulation 2, AdamW
at `2e-5`, weight decay `0.01`, warmup ratio `0.10`, and no early stopping. Every fold
started from the same authenticated base snapshot. Epoch checkpoints were ranked by
macro F1 at 0.50, validation loss, then earlier epoch.

### OOF results

Every seed covered all 299 official rows exactly once. The aggregate probability is the
per-row mean of the three complete OOF probability vectors.

| Seed | Threshold | Macro F1 | Label-0 F1 | Label-1 F1 | Accuracy | Confusion matrix | Predicted 0 / 1 |
|---:|---:|---:|---:|---:|---:|---|---:|
| 17 | 0.50 | 0.352814 | 0.000000 | 0.705628 | 0.545151 | `[[0,136],[0,163]]` | 0 / 299 |
| 29 | 0.50 | 0.510867 | 0.363636 | 0.658098 | 0.555184 | `[[38,98],[35,128]]` | 73 / 226 |
| 43 | 0.50 | 0.451182 | 0.263959 | 0.638404 | 0.515050 | `[[26,110],[35,128]]` | 61 / 238 |
| Aggregate | 0.50 | 0.389572 | 0.091503 | 0.687640 | 0.535117 | `[[7,129],[10,153]]` | 17 / 282 |

Threshold-0.50 seed macro F1: mean `0.438287522754545`, population standard deviation
`0.065165921463545`, minimum `0.352813852813853`, maximum `0.510867025005842`.
The aggregate prediction share was 94.31% label 1, so the baseline is not accepted at the
frozen primary threshold.

The frozen `0.20`–`0.80` grid selected `0.54` after averaging the repeated OOF
probabilities:

| Seed | Threshold | Macro F1 | Label-0 F1 | Label-1 F1 | Accuracy | Confusion matrix | Predicted 0 / 1 |
|---:|---:|---:|---:|---:|---:|---|---:|
| 17 | 0.54 | 0.571902 | 0.570470 | 0.573333 | 0.571906 | `[[85,51],[77,86]]` | 162 / 137 |
| 29 | 0.54 | 0.555164 | 0.552189 | 0.558140 | 0.555184 | `[[82,54],[79,84]]` | 161 / 138 |
| 43 | 0.54 | 0.538315 | 0.498168 | 0.578462 | 0.541806 | `[[68,68],[69,94]]` | 137 / 162 |
| Aggregate | 0.54 | 0.564979 | 0.575163 | 0.554795 | 0.565217 | `[[88,48],[82,81]]` | 170 / 129 |

The aggregate macro-F1 gain over 0.50 was `0.17540710091434858`. Selected-threshold seed
macro F1 had mean `0.5551268759488935`, population standard deviation
`0.013711675860275848`, minimum `0.5383150183150183`, and maximum
`0.5719015659955258`. Its maximum predicted-class share was 56.86%, so threshold 0.54
passes the collapse guard. It does not retroactively make the baseline acceptable at the
predeclared primary threshold.

Across the 15 fixed-threshold fold records, macro F1 had mean `0.4125223750478814`,
standard deviation `0.08442552148349124`, minimum `0.35106382978723405`, and maximum
`0.5622895622895623`.

### Fold metrics

`Rows` and `groups` show train/validation counts. Every overlap count was zero. A dash in
the null-context column means that fold contained no null-context validation row.

| Seed/fold | Rows | Groups | Selected epoch | Validation loss | Macro F1 | F1-0 | F1-1 | Accuracy | Confusion matrix | Context / null macro F1 | Seconds | Peak bytes | Checkpoint bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| 17/1 | 238/61 | 236/60 | 3 | 0.677486 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0.354839 / 0.000000 | 6.918 | 3461376512 | 444062405 |
| 17/2 | 240/59 | 237/59 | 3 | 0.682518 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0.351648 / — | 6.716 | 3471698432 | 444062405 |
| 17/3 | 239/60 | 237/59 | 3 | 0.677384 | 0.354839 | 0.000000 | 0.709677 | 0.550000 | `[[0,27],[0,33]]` | 0.354839 / — | 6.611 | 3458713088 | 444062405 |
| 17/4 | 239/60 | 237/59 | 2 | 0.679370 | 0.354839 | 0.000000 | 0.709677 | 0.550000 | `[[0,27],[0,33]]` | 0.351648 / 1.000000 | 6.414 | 3468235264 | 444062405 |
| 17/5 | 240/59 | 237/59 | 1 | 0.693244 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0.351648 / — | 6.126 | 3476185600 | 444062405 |
| 29/1 | 238/61 | 236/60 | 3 | 0.682233 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0.351064 / — | 6.636 | 3463367168 | 444062405 |
| 29/2 | 240/59 | 237/59 | 3 | 0.688009 | 0.542636 | 0.418605 | 0.666667 | 0.576271 | `[[9,18],[7,25]]` | 0.542636 / — | 6.616 | 3457182208 | 444062405 |
| 29/3 | 240/59 | 237/59 | 3 | 0.673522 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0.348315 / 1.000000 | 6.643 | 3466056192 | 444062405 |
| 29/4 | 239/60 | 237/59 | 2 | 0.672262 | 0.548872 | 0.526316 | 0.571429 | 0.550000 | `[[15,12],[15,18]]` | 0.548872 / — | 7.273 | 3480952832 | 444062405 |
| 29/5 | 239/60 | 237/59 | 3 | 0.683354 | 0.562290 | 0.518519 | 0.606061 | 0.566667 | `[[14,13],[13,20]]` | 0.553030 / 1.000000 | 7.576 | 3465845248 | 444062405 |
| 43/1 | 238/61 | 236/60 | 1 | 0.680464 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0.351064 / — | 6.898 | 3459635712 | 444062405 |
| 43/2 | 239/60 | 237/59 | 3 | 0.678524 | 0.354839 | 0.000000 | 0.709677 | 0.550000 | `[[0,27],[0,33]]` | 0.354839 / — | 6.759 | 3461900800 | 444062405 |
| 43/3 | 240/59 | 237/59 | 3 | 0.693732 | 0.449883 | 0.384615 | 0.515152 | 0.457627 | `[[10,17],[15,17]]` | 0.442308 / 1.000000 | 7.194 | 3471247872 | 444062405 |
| 43/4 | 239/60 | 237/59 | 3 | 0.680924 | 0.531250 | 0.500000 | 0.562500 | 0.533333 | `[[14,13],[15,18]]` | 0.540260 / 0.000000 | 7.164 | 3460459008 | 444062405 |
| 43/5 | 240/59 | 237/59 | 1 | 0.690959 | 0.380252 | 0.117647 | 0.642857 | 0.491525 | `[[2,25],[5,27]]` | 0.380252 / — | 6.601 | 3462556160 | 444062405 |

### Epoch losses

Each cell is `training loss / validation loss` for that epoch. Initial and final batch
losses are also retained in ignored fold metadata.

| Seed/fold | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| 17/1 | 0.691051 / 0.684042 | 0.682200 / 0.681268 | 0.673902 / 0.677486 |
| 17/2 | 0.695972 / 0.686391 | 0.687177 / 0.683006 | 0.675607 / 0.682518 |
| 17/3 | 0.693456 / 0.686369 | 0.680898 / 0.679264 | 0.675475 / 0.677384 |
| 17/4 | 0.689927 / 0.684424 | 0.680856 / 0.679370 | 0.679814 / 0.678068 |
| 17/5 | 0.690599 / 0.693244 | 0.680377 / 0.693748 | 0.670628 / 0.695333 |
| 29/1 | 0.693203 / 0.685355 | 0.683036 / 0.683137 | 0.677478 / 0.682233 |
| 29/2 | 0.688405 / 0.689064 | 0.680684 / 0.690752 | 0.670563 / 0.688009 |
| 29/3 | 0.693939 / 0.683718 | 0.686200 / 0.678007 | 0.675702 / 0.673522 |
| 29/4 | 0.689602 / 0.678589 | 0.676826 / 0.672262 | 0.672766 / 0.670089 |
| 29/5 | 0.686961 / 0.685881 | 0.679529 / 0.685465 | 0.670853 / 0.683354 |
| 43/1 | 0.692305 / 0.680464 | 0.690837 / 0.684802 | 0.680487 / 0.681457 |
| 43/2 | 0.690766 / 0.682389 | 0.689404 / 0.683407 | 0.679992 / 0.678524 |
| 43/3 | 0.685371 / 0.689023 | 0.682405 / 0.691671 | 0.673021 / 0.693732 |
| 43/4 | 0.692518 / 0.686995 | 0.683359 / 0.680599 | 0.681508 / 0.680924 |
| 43/5 | 0.689868 / 0.690959 | 0.681092 / 0.692048 | 0.668665 / 0.692759 |

### Integrity and outcome

- total runtime: `105.33692079999992` seconds;
- maximum per-fold allocated VRAM: `3480952832` bytes;
- selected checkpoints: 15, totaling `6660936075` bytes;
- failed or retried folds: none;
- group overlap: zero in every fold;
- checkpoint reload: exact predictions, metrics, and validation loss for all 15 folds;
- retained optimizer/scheduler states: none;
- legacy-at-run-time token audit: 299 rows, 297 context-present, two null-context, and no
  reported prompt/context/response truncation; the route counts are invalid for
  schema-aware V4 interpretation;
- classification: **failed baseline reproduction** because class collapse persisted at the
  predeclared primary threshold 0.50. The healthy 0.54 grid result is a diagnostic and is
  not permission to accept or deploy the candidate.

The repository-recorded V3 result is a `0.525` Kaggle public-leaderboard score from a
different evaluation set and protocol: length 512, four official epochs at `1e-5`, one
five-fold assignment, and deployed threshold 0.54. It is not directly compared with these
official-only repeated grouped OOF scores.

## Schema-corrected V4 baseline

Interpretation: **schema-aware V4 baseline isolating only corrected null-context routing**.
The run used the authenticated 299-row official sample, seeds 17/29/43, five grouped folds
per seed, three epochs, length 256, batch 8, accumulation 2, AdamW at `2e-5`, weight decay
`0.01`, warmup `0.10`, and CUDA fp16. No public or competition-test data was used.

Preprocessing and OOF coverage were exact for every seed: 130 context-present rows
(47 label 0, 83 label 1) and 169 context-absent rows (89 label 0, 80 label 1). Every
validation fold contained both labels, every train/validation group overlap count was
zero, and all 15 selected checkpoints reproduced probabilities, predictions, metrics,
and validation loss exactly after offline reload.

### Aggregate and seed results

| Scope | Threshold | Macro F1 | F1-0 | F1-1 | Accuracy | Confusion matrix | Predicted 0 / 1 |
|---|---:|---:|---:|---:|---:|---|---:|
| Seed 17 | 0.50 | 0.391149 | 0.081633 | 0.700665 | 0.548495 | `[[6,130],[5,158]]` | 11 / 288 |
| Seed 29 | 0.50 | 0.516056 | 0.367150 | 0.664962 | 0.561873 | `[[38,98],[33,130]]` | 71 / 228 |
| Seed 43 | 0.50 | 0.470674 | 0.281250 | 0.660099 | 0.538462 | `[[27,109],[29,134]]` | 56 / 243 |
| Aggregate | 0.50 | 0.386019 | 0.090323 | 0.681716 | 0.528428 | `[[7,129],[12,151]]` | 19 / 280 |
| Seed 17 | 0.54 | 0.564394 | 0.545455 | 0.583333 | 0.565217 | `[[78,58],[72,91]]` | 150 / 149 |
| Seed 29 | 0.54 | 0.555164 | 0.552189 | 0.558140 | 0.555184 | `[[82,54],[79,84]]` | 161 / 138 |
| Seed 43 | 0.54 | 0.538043 | 0.524138 | 0.551948 | 0.538462 | `[[76,60],[78,85]]` | 154 / 145 |
| Aggregate | 0.54 | 0.565096 | 0.572368 | 0.557823 | 0.565217 | `[[87,49],[81,82]]` | 168 / 131 |

At threshold 0.50, seed macro F1 mean/std/min/max were
`0.459292959830255` / `0.051624130281861` / `0.391148920765646` /
`0.516055697641375`. The aggregate predicted-label-1 share was 93.65%, so the primary
collapse guard failed. The frozen global grid selected 0.54, with macro-F1 gain
`0.17907669701920176`; its maximum predicted-class share was 56.19%, so the selected
threshold passed the guard. Selected-threshold seed macro F1 mean/std/min/max were
`0.5525336581404477` / `0.010917335124767492` / `0.5380429914912674` /
`0.5643939393939394`.

Diagnostic-only route optima were 0.58 for context-present rows (macro F1 `0.541772`,
confusion matrix `[[23,24],[33,50]]`, predictions 56/74) and 0.52 for context-absent
rows (macro F1 `0.450000`, confusion matrix `[[60,29],[59,21]]`, predictions 119/50).
Both diagnostic thresholds passed the 90% balance guard; neither was selected or deployed.

Across the 15 fixed-0.50 fold records, macro F1 mean/std/min/max were
`0.43197022105314475` / `0.09012752886944012` / `0.35106382978723405` /
`0.5785714285714285`. Mean context-present and context-absent macro F1 were
`0.3888662133652888` and `0.38571390379768083`, respectively.

### Fold splits and routes

`P/A` means context-present/context-absent. Label counts are label 0 / label 1.

| Seed/fold | Train rows/groups | Train labels | Train P/A | Validation rows/groups | Validation labels | Validation P/A |
|---|---:|---:|---:|---:|---:|---:|
| 17/1 | 238 / 236 | 108 / 130 | 105 / 133 | 61 / 60 | 28 / 33 | 25 / 36 |
| 17/2 | 240 / 237 | 109 / 131 | 103 / 137 | 59 / 59 | 27 / 32 | 27 / 32 |
| 17/3 | 239 / 237 | 109 / 130 | 104 / 135 | 60 / 59 | 27 / 33 | 26 / 34 |
| 17/4 | 239 / 237 | 109 / 130 | 103 / 136 | 60 / 59 | 27 / 33 | 27 / 33 |
| 17/5 | 240 / 237 | 109 / 131 | 105 / 135 | 59 / 59 | 27 / 32 | 25 / 34 |
| 29/1 | 238 / 236 | 108 / 130 | 103 / 135 | 61 / 60 | 28 / 33 | 27 / 34 |
| 29/2 | 240 / 237 | 109 / 131 | 106 / 134 | 59 / 59 | 27 / 32 | 24 / 35 |
| 29/3 | 240 / 237 | 109 / 131 | 102 / 138 | 59 / 59 | 27 / 32 | 28 / 31 |
| 29/4 | 239 / 237 | 109 / 130 | 106 / 133 | 60 / 59 | 27 / 33 | 24 / 36 |
| 29/5 | 239 / 237 | 109 / 130 | 103 / 136 | 60 / 59 | 27 / 33 | 27 / 33 |
| 43/1 | 238 / 236 | 108 / 130 | 102 / 136 | 61 / 60 | 28 / 33 | 28 / 33 |
| 43/2 | 239 / 237 | 109 / 130 | 106 / 133 | 60 / 59 | 27 / 33 | 24 / 36 |
| 43/3 | 240 / 237 | 109 / 131 | 102 / 138 | 59 / 59 | 27 / 32 | 28 / 31 |
| 43/4 | 239 / 237 | 109 / 130 | 103 / 136 | 60 / 59 | 27 / 33 | 27 / 33 |
| 43/5 | 240 / 237 | 109 / 131 | 107 / 133 | 59 / 59 | 27 / 32 | 23 / 36 |

### Fold metrics at threshold 0.50

| Seed/fold | Epoch | Val loss | Macro F1 | F1-0 | F1-1 | Acc. | Confusion matrix | Pred. 0/1 | Present macro / CM | Absent macro / CM | Seconds | Peak bytes |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|
| 17/1 | 3 | 0.672343 | 0.487726 | 0.277778 | 0.697674 | 0.573770 | `[[5,23],[3,30]]` | 8/53 | 0.404762 / `[[0,8],[0,17]]` | 0.474026 / `[[5,15],[3,13]]` | 6.852 | 3461376512 |
| 17/2 | 3 | 0.682808 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0/59 | 0.386364 / `[[0,10],[0,17]]` | 0.319149 / `[[0,17],[0,15]]` | 6.807 | 3471698432 |
| 17/3 | 3 | 0.678442 | 0.354839 | 0.000000 | 0.709677 | 0.550000 | `[[0,27],[0,33]]` | 0/60 | 0.395349 / `[[0,9],[0,17]]` | 0.320000 / `[[0,18],[0,16]]` | 6.738 | 3458713088 |
| 17/4 | 3 | 0.678215 | 0.377778 | 0.066667 | 0.688889 | 0.533333 | `[[1,26],[2,31]]` | 3/57 | 0.400000 / `[[0,9],[0,18]]` | 0.336508 / `[[1,17],[2,13]]` | 6.769 | 3468235264 |
| 17/5 | 1 | 0.692813 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0/59 | 0.358974 / `[[0,11],[0,14]]` | 0.346154 / `[[0,16],[0,18]]` | 6.245 | 3476185600 |
| 29/1 | 3 | 0.682729 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0/61 | 0.400000 / `[[0,9],[0,18]]` | 0.306122 / `[[0,19],[0,15]]` | 6.911 | 3463367168 |
| 29/2 | 2 | 0.687566 | 0.578571 | 0.500000 | 0.657143 | 0.593220 | `[[12,15],[9,23]]` | 21/38 | 0.368421 / `[[0,10],[0,14]]` | 0.597039 / `[[12,5],[9,9]]` | 6.646 | 3457182208 |
| 29/3 | 3 | 0.673456 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0/59 | 0.404255 / `[[0,9],[0,19]]` | 0.295455 / `[[0,18],[0,13]]` | 6.860 | 3466056192 |
| 29/4 | 2 | 0.673324 | 0.562290 | 0.518519 | 0.606061 | 0.566667 | `[[14,13],[13,20]]` | 27/33 | 0.400000 / `[[0,8],[0,16]]` | 0.458194 / `[[14,5],[13,4]]` | 6.711 | 3480952832 |
| 29/5 | 3 | 0.683459 | 0.554286 | 0.480000 | 0.628571 | 0.566667 | `[[12,15],[11,22]]` | 23/37 | 0.372093 / `[[0,11],[0,16]]` | 0.529915 / `[[12,4],[11,6]]` | 6.981 | 3465845248 |
| 43/1 | 1 | 0.681465 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0/61 | 0.404255 / `[[0,9],[0,19]]` | 0.297872 / `[[0,19],[0,14]]` | 6.378 | 3459635712 |
| 43/2 | 3 | 0.678703 | 0.354839 | 0.000000 | 0.709677 | 0.550000 | `[[0,27],[0,33]]` | 0/60 | 0.428571 / `[[0,6],[0,18]]` | 0.294118 / `[[0,21],[0,15]]` | 6.877 | 3460852224 |
| 43/3 | 3 | 0.693107 | 0.493937 | 0.408163 | 0.579710 | 0.508475 | `[[10,17],[12,20]]` | 22/37 | 0.348837 / `[[0,13],[0,15]]` | 0.470085 / `[[10,4],[12,5]]` | 6.738 | 3471247872 |
| 43/4 | 3 | 0.681470 | 0.548872 | 0.526316 | 0.571429 | 0.550000 | `[[15,12],[15,18]]` | 30/30 | 0.400000 / `[[0,9],[0,18]]` | 0.312500 / `[[15,3],[15,0]]` | 6.836 | 3460459008 |
| 43/5 | 3 | 0.692590 | 0.409344 | 0.129032 | 0.689655 | 0.542373 | `[[2,25],[2,30]]` | 4/55 | 0.361111 / `[[0,10],[0,13]]` | 0.428571 / `[[2,15],[2,17]]` | 6.655 | 3462556160 |

### Epoch losses

Each cell is mean training loss / validation loss.

| Seed/fold | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| 17/1 | 0.694837 / 0.682501 | 0.676865 / 0.675485 | 0.668835 / 0.672343 |
| 17/2 | 0.695903 / 0.686780 | 0.686766 / 0.683370 | 0.675086 / 0.682808 |
| 17/3 | 0.693630 / 0.685465 | 0.681704 / 0.679704 | 0.676581 / 0.678442 |
| 17/4 | 0.689210 / 0.684115 | 0.681141 / 0.679378 | 0.680444 / 0.678215 |
| 17/5 | 0.690210 / 0.692813 | 0.680195 / 0.693541 | 0.668529 / 0.694783 |
| 29/1 | 0.693482 / 0.685715 | 0.682123 / 0.683426 | 0.677715 / 0.682729 |
| 29/2 | 0.689510 / 0.690380 | 0.678150 / 0.687566 | 0.671613 / 0.686271 |
| 29/3 | 0.693435 / 0.683072 | 0.685632 / 0.676932 | 0.675798 / 0.673456 |
| 29/4 | 0.689899 / 0.678670 | 0.677066 / 0.673324 | 0.672612 / 0.671517 |
| 29/5 | 0.687148 / 0.686239 | 0.678619 / 0.685291 | 0.671745 / 0.683459 |
| 43/1 | 0.691637 / 0.681465 | 0.691525 / 0.685147 | 0.679581 / 0.681553 |
| 43/2 | 0.690619 / 0.685539 | 0.689135 / 0.684172 | 0.676729 / 0.678703 |
| 43/3 | 0.685458 / 0.688766 | 0.681875 / 0.691506 | 0.672478 / 0.693107 |
| 43/4 | 0.692580 / 0.686222 | 0.682200 / 0.680868 | 0.680697 / 0.681470 |
| 43/5 | 0.690147 / 0.690306 | 0.682187 / 0.691700 | 0.669645 / 0.692590 |

### Checkpoint integrity and outcome

Every selected checkpoint was `444062405` bytes and passed exact probability,
prediction, metric, and validation-loss reload comparison with zero missing or unexpected
keys. Weight SHA-256 values by seed/fold were:

| Seed/fold | SHA-256 | Retained |
|---|---|---|
| 17/1 | `a7396e82661ec984ba4a10fbf4ee587547edc2bcf50b62fb243270b356e4e93b` | yes |
| 17/2 | `a8442d6b6b7dbc325eeee958c757b7f88c9c6d046146c1da5356c2fc01137dad` | no |
| 17/3 | `1f54f179f546396f2f3043af64e8dc73ebe3638e4e68f43fd62d5a409c8727c4` | no |
| 17/4 | `9e5447ebb2d3c23585f9faefaba0bf73aa4cc738b612fcfa23e95fe129b0922c` | no |
| 17/5 | `110bdd61b016c553fc84fb3942b24a7e60818321f9cb2e6de1e5c3f80ee9f7f2` | no |
| 29/1 | `71f244c2f451b05f16914af4e420bb4823e608ee35cc36c442bdd7595c7f04c9` | no |
| 29/2 | `7afb7cf83737c5e86bb58a59820158516639357e073eb609aac305e361292d9a` | no |
| 29/3 | `00578d0d4a1eed2286af0cad303fab1f81fd73adca74472895f64c31c4f266f9` | no |
| 29/4 | `074adef9755e5920dfb916ab297076544287ce664314590b160a2494cbaa9de7` | no |
| 29/5 | `6fb0d207ac7424a5a8afb98f4aaf467a32bd45ef45bdc0d94f9f9dcbbbcc7748` | no |
| 43/1 | `0c412b0b9458392ad8c2cdc1ca0f3a62f5109e7078efdd2942cc3b619941f40b` | no |
| 43/2 | `607230ce770f53a1052327cdbdf1b0e9bf4cef5804c43b90a26aabcddfbcd15b` | no |
| 43/3 | `310cb4cd82b29f33ce05145bb5f5bc95beaed08248c59d6c8e48086972ea906f` | no |
| 43/4 | `c152b19b616cd56844992aaaf687f4eeab6e0a69fd83df63ebcbda8552e4acd4` | no |
| 43/5 | `27a981f718663c8b96a1af33a6709d4c225c47027748fd33dbbe22687909e767` | no |

Total wall-clock runtime was `109.50121030000082` seconds and maximum allocated VRAM was
`3480952832` bytes. All 15 selected checkpoints totaled `6660936075` transient bytes;
compact retention left one representative checkpoint of `444062405` bytes plus aggregate
JSON metadata. No optimizer or scheduler state was retained, and no row-level probability
array was persisted.

Classification: **failed schema-corrected baseline** because threshold 0.50 exceeded the
90% predicted-class guard. The balanced threshold-0.54 result and route-specific optima
remain diagnostic and do not authorize deployment or another candidate.

## Schema-corrected historical-schedule control

Interpretation: **Schema-corrected V4 preprocessing with the historical V3 training
schedule.** This is not an exact historical V3 reproduction. It used the identical
authenticated sample, corrected 130/169 routing, seeds, folds, grouping, serialization,
batch size, accumulation, weight decay, warmup, and threshold procedure as the corrected
baseline. Only maximum length 512, four epochs, learning rate `1e-5`, and mandatory final
epoch-4 checkpoint selection differed.

All three seeds covered every official row exactly once. Every fold had zero group
overlap, both validation labels, finite losses, final epoch 4 selected, and exact offline
reload equality for probabilities, predictions, metrics, and validation loss. The split
and route table in the preceding corrected-baseline section applies unchanged.

### Aggregate and seed results

| Scope | Threshold | Macro F1 | F1-0 | F1-1 | Accuracy | Confusion matrix | Predicted 0 / 1 |
|---|---:|---:|---:|---:|---:|---|---:|
| Seed 17 | 0.50 | 0.373109 | 0.054422 | 0.691796 | 0.535117 | `[[4,132],[7,156]]` | 11 / 288 |
| Seed 29 | 0.50 | 0.411216 | 0.137500 | 0.684932 | 0.538462 | `[[11,125],[13,150]]` | 24 / 275 |
| Seed 43 | 0.50 | 0.466608 | 0.290000 | 0.643216 | 0.525084 | `[[29,107],[35,128]]` | 64 / 235 |
| Aggregate | 0.50 | 0.375394 | 0.066225 | 0.684564 | 0.528428 | `[[5,131],[10,153]]` | 15 / 284 |
| Seed 17 | 0.52 | 0.577096 | 0.531835 | 0.622356 | 0.581940 | `[[71,65],[60,103]]` | 131 / 168 |
| Seed 29 | 0.52 | 0.551252 | 0.482213 | 0.620290 | 0.561873 | `[[61,75],[56,107]]` | 117 / 182 |
| Seed 43 | 0.52 | 0.546037 | 0.512635 | 0.579439 | 0.548495 | `[[71,65],[70,93]]` | 141 / 158 |
| Aggregate | 0.52 | 0.559904 | 0.530466 | 0.589342 | 0.561873 | `[[74,62],[69,94]]` | 143 / 156 |

At threshold 0.50, seed macro F1 mean/std/min/max were
`0.416977560804665` / `0.0383876859329981` / `0.373108888788331` /
`0.466608040201005`. The aggregate predicted-label-1 share was 94.98%, so the primary
collapse guard failed. The frozen grid selected 0.52, with macro-F1 gain
`0.1845093593292908`; its maximum predicted-class share was 52.17%, so the selected
threshold passed the balance guard. Selected-threshold seed macro F1 mean/std/min/max
were `0.5581282711110416` / `0.013579988216513543` / `0.5460373156989102` /
`0.5770958507303936`.

At the selected global threshold, context-present macro F1 was `0.386792` and
context-absent macro F1 was `0.432496`. Diagnostic-only route optima were 0.56 for
context-present rows (macro F1 `0.513499`, confusion matrix `[[15,32],[24,59]]`) and
0.52 for context-absent rows (macro F1 `0.432496`, confusion matrix
`[[74,15],[68,12]]`). Neither route threshold was deployed.

### Direct corrected-baseline comparison

| Metric | Corrected baseline | Historical-schedule control | Delta |
|---|---:|---:|---:|
| Selected global threshold | 0.54 | 0.52 | -0.02 |
| Selected-threshold macro F1 | 0.565096 | 0.559904 | -0.005192 |
| Threshold-0.50 macro F1 | 0.386019 | 0.375394 | -0.010625 |
| Context-present diagnostic macro F1 | 0.541772 | 0.513499 | -0.028273 |
| Context-absent diagnostic macro F1 | 0.450000 | 0.432496 | -0.017504 |

The predeclared aggregate verdict is **approximately equivalent** because the selected
macro-F1 change is between -0.01 and +0.01. Threshold-0.50 and context-absent changes are
inconclusive under the predeclared bands; the context-present diagnostic degraded by more
than 0.02. Both aggregate threshold-0.50 results remain collapsed.

Across the 15 final-epoch fold records, macro F1 mean/std/min/max were
`0.4025559760304357` / `0.06444606550192887` / `0.32977624784853704` /
`0.5506349723217193`. Mean context-present and context-absent fold macro F1 were
`0.3888662133652888` and `0.36487440948664845`.

### Fold metrics at threshold 0.50

| Seed/fold | Val loss | Macro F1 | F1-0 | F1-1 | Acc. | Confusion matrix | Pred. 0/1 | Present / absent macro F1 | Seconds | Peak bytes |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 17/1 | 0.679976 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0/61 | 0.404762 / 0.307692 | 10.312 | 6014071296 |
| 17/2 | 0.687947 | 0.329776 | 0.057143 | 0.602410 | 0.440678 | `[[1,26],[7,25]]` | 8/51 | 0.386364 / 0.245128 | 9.945 | 6016288768 |
| 17/3 | 0.681453 | 0.354839 | 0.000000 | 0.709677 | 0.550000 | `[[0,27],[0,33]]` | 0/60 | 0.395349 / 0.320000 | 9.992 | 5992645632 |
| 17/4 | 0.680477 | 0.394410 | 0.071429 | 0.717391 | 0.566667 | `[[1,26],[0,33]]` | 1/59 | 0.400000 / 0.371781 | 10.467 | 6011762176 |
| 17/5 | 0.690777 | 0.428516 | 0.137931 | 0.719101 | 0.576271 | `[[2,25],[0,32]]` | 2/57 | 0.358974 / 0.471111 | 9.801 | 6033473024 |
| 29/1 | 0.682985 | 0.344086 | 0.000000 | 0.688172 | 0.524590 | `[[0,28],[1,32]]` | 1/60 | 0.400000 / 0.291667 | 10.355 | 6009752064 |
| 29/2 | 0.688688 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0/59 | 0.368421 / 0.339623 | 10.103 | 6016314880 |
| 29/3 | 0.672810 | 0.351648 | 0.000000 | 0.703297 | 0.542373 | `[[0,27],[0,32]]` | 0/59 | 0.404255 / 0.295455 | 10.464 | 6028749312 |
| 29/4 | 0.677010 | 0.400000 | 0.200000 | 0.600000 | 0.466667 | `[[4,23],[9,24]]` | 13/47 | 0.400000 / 0.325000 | 10.117 | 6023953408 |
| 29/5 | 0.682739 | 0.550635 | 0.378378 | 0.722892 | 0.616667 | `[[7,20],[3,30]]` | 10/50 | 0.372093 / 0.619231 | 9.729 | 5998017024 |
| 43/1 | 0.677502 | 0.351064 | 0.000000 | 0.702128 | 0.540984 | `[[0,28],[0,33]]` | 0/61 | 0.404255 / 0.297872 | 10.359 | 6004443648 |
| 43/2 | 0.681852 | 0.394410 | 0.071429 | 0.717391 | 0.566667 | `[[1,26],[0,33]]` | 1/59 | 0.428571 / 0.345455 | 9.954 | 6005763072 |
| 43/3 | 0.692888 | 0.464735 | 0.392157 | 0.537313 | 0.474576 | `[[10,17],[14,18]]` | 24/35 | 0.348837 / 0.388158 | 9.810 | 6005369856 |
| 43/4 | 0.681226 | 0.499358 | 0.315789 | 0.682927 | 0.566667 | `[[6,21],[5,28]]` | 11/49 | 0.400000 / 0.477167 | 10.349 | 6023477760 |
| 43/5 | 0.693103 | 0.472150 | 0.436364 | 0.507937 | 0.474576 | `[[12,15],[16,16]]` | 28/31 | 0.361111 / 0.377778 | 10.225 | 6006544896 |

### Epoch losses

Each cell is mean training loss / validation loss. Epoch 4 was retained for every fold.

| Seed/fold | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---:|---:|---:|---:|
| 17/1 | 0.696182 / 0.684170 | 0.685225 / 0.683730 | 0.681109 / 0.680064 | 0.678451 / 0.679976 |
| 17/2 | 0.695603 / 0.690918 | 0.685268 / 0.689014 | 0.678886 / 0.688700 | 0.674597 / 0.687947 |
| 17/3 | 0.699518 / 0.689559 | 0.685052 / 0.685205 | 0.683408 / 0.682902 | 0.682446 / 0.681453 |
| 17/4 | 0.694457 / 0.687370 | 0.690132 / 0.683228 | 0.682824 / 0.681535 | 0.680508 / 0.680477 |
| 17/5 | 0.696503 / 0.689039 | 0.685876 / 0.689213 | 0.677418 / 0.690339 | 0.676961 / 0.690777 |
| 29/1 | 0.695410 / 0.689053 | 0.685842 / 0.687372 | 0.677737 / 0.683778 | 0.676514 / 0.682985 |
| 29/2 | 0.689972 / 0.690728 | 0.682568 / 0.691530 | 0.679624 / 0.689230 | 0.670267 / 0.688688 |
| 29/3 | 0.690133 / 0.684008 | 0.685783 / 0.678107 | 0.680488 / 0.673340 | 0.674233 / 0.672810 |
| 29/4 | 0.692675 / 0.684310 | 0.682202 / 0.679875 | 0.680869 / 0.677743 | 0.676342 / 0.677010 |
| 29/5 | 0.690651 / 0.686174 | 0.684680 / 0.685441 | 0.678081 / 0.683089 | 0.671143 / 0.682739 |
| 43/1 | 0.691798 / 0.682633 | 0.691097 / 0.684506 | 0.680915 / 0.679567 | 0.678878 / 0.677502 |
| 43/2 | 0.687891 / 0.687996 | 0.684246 / 0.688403 | 0.680942 / 0.683610 | 0.676315 / 0.681852 |
| 43/3 | 0.688656 / 0.689453 | 0.678772 / 0.690554 | 0.674107 / 0.692962 | 0.672149 / 0.692888 |
| 43/4 | 0.691537 / 0.689266 | 0.687148 / 0.683643 | 0.681816 / 0.684456 | 0.674477 / 0.681226 |
| 43/5 | 0.691880 / 0.690066 | 0.681270 / 0.690380 | 0.672742 / 0.692135 | 0.668531 / 0.693103 |

### Checkpoint integrity and outcome

Every final checkpoint was `444062405` bytes. All 15 digests and split distributions are
retained in ignored aggregate/fold JSON metadata. Exact reload passed 15/15 with zero
missing or unexpected keys. Compact retention left only seed 17/fold 1, whose weight
SHA-256 is `7f5de760ca2b30e040b6818cee59b370484be11c46d302825a500907a3d5183b`.

Total wall-clock runtime was `160.57824299999993` seconds and maximum allocated VRAM was
`6033473024` bytes. All selected checkpoints totaled `6660936075` transient bytes;
retained checkpoint storage was `444062405` bytes plus aggregate JSON metadata. No
optimizer state, scheduler state, or row-level probability array was persisted.

Classification: **failed control at threshold 0.50** because it exceeded the 90% class
guard. Relative to the corrected baseline, the historical schedule is **approximately
equivalent** at the selected global threshold and does not justify adoption.

## Candidate comparison

Not run. No candidate or threshold is selected.

## Safety record

- data admitted: authenticated official labeled sample only;
- public 5K: excluded;
- public 20K: unavailable and excluded;
- competition test: not accessed;
- external API: not used;
- model snapshot: exact approved revision downloaded to an ignored local directory and
  authenticated before use;
- training/checkpoint: one two-step smoke, the authorized legacy-compatible reproduction,
  the authorized schema-corrected 15-fit baseline, and the authorized schema-corrected
  historical-schedule control; every selected checkpoint is confined to ignored
  `artifacts/v4/`;
- prediction/submission: no competition-test prediction or submission was produced.
