# Version 4 Experiment Log

## Status

Gate 4 GPU smoke training passed for the V3-compatible candidate. This was one diagnostic
run only; it is not a candidate-selection result. No submission or competition-test
inference exists.

## Frozen family

| Order | Configuration | Result status |
|---:|---|---|
| 1 | V3-compatible serialization, comparison encoder | One two-step smoke run passed |
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
- aggregate token audit over 299 official rows: 297 context-present and two null-context;
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

Not authorized or run. This section remains empty until the smoke gate passes and the
reproduction is separately reviewed.

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
- training/checkpoint: one bounded two-step smoke run only, with its checkpoint confined
  to ignored `artifacts/v4/`;
- prediction/submission: no competition-test prediction or submission was produced.
