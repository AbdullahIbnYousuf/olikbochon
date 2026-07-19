# Paper-Ready V7 Tables

## Main and route results

| Model | Macro F1 | F1 hallucinated | F1 faithful | Present | Absent |
|---|---:|---:|---:|---:|---:|
| E0 V4-A retrieval | 0.688963 | 0.688963 | 0.688963 | 0.895745 | 0.495833 |
| E1 exact + Candidate I | **0.704575** | 0.686461 | **0.722689** | **0.900026** | **0.545327** |
| E2 evidence-conditioned frozen BanglaBERT | 0.523985 | 0.484185 | 0.563786 | 0.567727 | 0.476658 |
| E3 deterministic route blend | 0.700955 | **0.691954** | 0.709957 | **0.900026** | 0.530421 |
| E4 nested logistic fusion | 0.681964 | 0.691892 | 0.672037 | **0.900026** | 0.470797 |

## Declared ablations

| Comparison | Result |
|---|---|
| V4-A retrieval versus exact/sparse routing | E1 +0.015612 macro F1; absent route +0.049494 |
| Lexical routing versus semantic verifier | E1 exceeds E2 by 0.180589 |
| Deterministic route blend versus nested fusion | E3 exceeds E4 by 0.018991 |
| Threshold 0.50 versus inner-selected | Fold-specific inner selections are retained in each `fold_metrics.csv`; untouched outer predictions were never tuned directly |
| Optional NLI | Not started after hard stop |

## Non-comparable historical estimates

| Estimate | Macro F1 | Caveat |
|---|---:|---|
| Historical V4-A | 0.687270 | 296-group protocol |
| Research Candidate I | 0.692051 | Different/non-fully-nested protocol |
| V7 E0 | 0.688963 | Strengthened 291-group repeated nested protocol |

Suggested framing: “We study Bengali hallucination detection under mixed evidence availability
and severe label scarcity. We evaluate a route-aware hybrid combining official or retrieved
evidence, lexical/numeric support, sparse modeling, and evidence-conditioned semantic
verification under duplicate-aware nested cross-fitting.”
