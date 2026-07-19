# V7 Experiment Summary

## Common-fold repeated OOF results

| Experiment | Status | Macro F1 | F1-0 | F1-1 | Present | Absent | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 V4-A | COMPLETE | 0.688963 | 0.688963 | 0.688963 | 0.895745 | 0.495833 | 0.194195 |
| E1 exact + Candidate I | COMPLETE | **0.704575** | 0.686461 | 0.722689 | **0.900026** | **0.545327** | 0.196674 |
| E2 frozen semantic | COMPLETE, negative ablation | 0.523985 | 0.484185 | 0.563786 | 0.567727 | 0.476658 | 0.372273 |
| E3 route blend | COMPLETE | 0.700955 | **0.691954** | 0.709957 | **0.900026** | 0.530421 | 0.196119 |
| E4 nested stack | REJECTED BY GATE | 0.681964 | 0.691892 | 0.672037 | **0.900026** | 0.470797 | **0.186103** |

E4 versus E0 is `-0.006999` macro F1. Seed deltas are `-0.029352`, `+0.017084`, and
`-0.008395`; only one seed improves. E4 has no class collapse (maximum class share `0.576366`),
but it fails the required improvement and stability conditions.

E1 improves E0 by `+0.015612` overall and `+0.049494` on context-absent rows. Its seed scores
are `0.680846`, `0.724120`, and `0.708561`; E0 seed scores are `0.711985`, `0.668362`, and
`0.683747`. This is useful ablation evidence, but the specified hard stop retains V4-A because
E4 itself did not produce an honest improvement.

## Earlier estimates (not comparable)

| System | Estimate | Protocol caveat |
|---|---:|---|
| Historical V4-A | 0.687270 | 296-group, one outer seed; not V7 common folds |
| Research Candidate I | 0.692051 | Different threshold/nesting protocol |
| V4-A Kaggle public | 0.685 | Supplied leaderboard evidence, not local CV |
