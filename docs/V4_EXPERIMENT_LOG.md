# Version 4 Experiment Log

## Status

Gate 3 foundation only. No model was downloaded, trained, evaluated, or used for
prediction. No checkpoint, submission, or competition-test inference exists.

## Frozen family

| Order | Configuration | Result status |
|---:|---|---|
| 1 | V3-compatible serialization, comparison encoder | Not run |
| 2 | Structured serialization, comparison encoder | Not run |
| 3 | Routed serialization, comparison encoder | Not run |
| 4 | Selected serialization, field-aware encoder | Not run |

Validation seeds are 17, 29, and 43 with five group-isolated folds per seed. The primary
metric is ordinary two-class macro F1. The threshold grid is 0.20–0.80 inclusive by 0.02.
The complete precommitted configuration and tie-break rules are in `V4_MODELING_PLAN.md`.

## Smoke test

Not authorized or run.

- timestamp: pending
- Git commit: pending
- configuration: pending
- seed: pending
- authenticated model snapshot role: pending
- training rows: pending
- validation groups: pending
- CUDA used: pending
- training loss: pending
- evaluation pipeline result: pending
- checkpoint reload result: pending
- runtime: pending
- peak VRAM: pending
- checkpoint size: pending
- warnings/failures: pending

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
- model download: not performed;
- training/checkpoint/prediction/submission: not performed.
