# Version 4 Experiment Log

## Status

Gate 4 GPU smoke training passed. Gate 5 completed all 15 authorized fits but the
V3-compatible baseline failed the frozen threshold-0.50 class-collapse acceptance guard.
No submission or competition-test inference exists.

Route audit correction (2026-07-18): the authenticated sample contains 167 literal
`[NULL]` context sentinels and two actual nulls. The completed smoke and reproduction
runs inherited V3's truthy-string behavior and therefore encoded those 167 sentinels as
context-present. Their saved losses, predictions, overall metrics, reload checks, and
calibration statistics remain exact records of the pipeline that ran, and their grouped
fold indices remain unchanged. They are not valid measurements of the corrected,
schema-aware V4 preprocessing path. All recorded context-present/null subgroup metrics
and the 297/2 truncation route counts are invalid for V4 route interpretation. Corrected
V4 counts are 130 context-present and 169 context-absent. No artifact was deleted and no
training was rerun.

## Frozen family

| Order | Configuration | Result status |
|---:|---|---|
| 1 | V3-compatible serialization, comparison encoder | Smoke passed; Gate 5 reproduction failed the 0.50 collapse guard |
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
- training/checkpoint: one two-step smoke plus the single authorized 15-fit reproduction;
  every selected checkpoint is confined to ignored `artifacts/v4/`;
- prediction/submission: no competition-test prediction or submission was produced.
