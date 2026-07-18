# V10 Nested Lexical-Semantic Ensemble Results

Status: **complete — Candidates U, V, and W rejected**.

The single frozen execution used the authenticated official 299 rows, corrected 130/169
routing, strengthened grouped outer folds for seeds 17, 29, and 43, and three genuine grouped
inner OOF folds inside every outer-training partition. The fixed decision threshold was 0.50.

## Base complementarity

Candidate I reproduced its frozen V5 null macro F1 of 0.534500467 and routed macro F1 of
0.692051375. Candidate R reproduced its V9 null macro F1 of 0.538316055 and routed macro F1 of
0.693786982.

Across 507 repeated null-route OOF evaluations, Candidate I and R probabilities had Pearson
correlation 0.125298 and their threshold-0.50 decisions disagreed on 45.759% of rows. They were
both correct on 156 evaluations, only I was correct on 115, only R was correct on 117, and both
were wrong on 119. This establishes substantial but approximately symmetric complementarity;
the nested ensembles did not convert it into a qualifying aggregate improvement.

## Aggregate results

| Candidate | Null macro F1 | F1 label 0 / 1 | Null accuracy | Null confusion | Null counts 0 / 1 | Brier | Routed macro F1 | Routed F1 label 0 / 1 | Routed accuracy | Routed confusion | Routed counts 0 / 1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| U — fixed 50/50 average | 0.538445 | 0.535714 / 0.541176 | 0.538462 | `[[135,132],[102,138]]` | 237 / 270 | 0.251192 | **0.694316** | 0.656489 / 0.732143 | 0.698997 | `[[258,150],[120,369]]` | 378 / 519 |
| V — nested convex blend | 0.528592 | 0.530452 / 0.526733 | 0.528600 | `[[135,132],[107,133]]` | 242 / 265 | 0.260067 | 0.689081 | 0.652339 / 0.725823 | 0.693423 | `[[258,150],[125,364]]` | 383 / 514 |
| W — nested logistic stacker | 0.508807 | 0.502994 / 0.514620 | 0.508876 | `[[126,141],[108,132]]` | 234 / 273 | 0.253988 | 0.677058 | 0.636015 / 0.718101 | 0.682274 | `[[249,159],[126,363]]` | 375 / 522 |

The context-present route remained fixed at 0.900026 macro F1. Routed seed standard deviations
were 0.010492 for U, 0.007466 for V, and 0.009317 for W; all passed stability and class-collapse
guards.

## Per-seed paired robustness

| Seed | Candidate I | Candidate R | Candidate U | Candidate V | Candidate W |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.698442 | 0.700592 | **0.703728** | 0.680178 | 0.689526 |
| 29 | 0.683857 | 0.672149 | 0.679594 | **0.698442** | 0.674469 |
| 43 | 0.693787 | **0.708379** | 0.699314 | 0.688513 | 0.667144 |

Paired routed changes versus Candidate I were:

- U: +0.005286, -0.004263, +0.005527; improvement in two of three seeds.
- V: -0.018264, +0.014585, -0.005274; improvement in one seed.
- W: -0.008916, -0.009388, -0.026643; improvement in zero seeds.

U satisfies the two-seed paired condition but its aggregate 0.694316 is below the required
0.697051. V and W fail both the aggregate and paired-seed conditions. No candidate is eligible.

Candidate V selected Candidate-I weight 0.00 in 3 folds, 0.25 in 7 folds, 0.50 in 1 fold, and
1.00 in 4 folds; weight 0.75 was never selected. Candidate W standardized coefficients varied
substantially: Candidate I mean 0.095189 with range -0.281806 to 0.378364, and Candidate R mean
0.212189 with range -0.298361 to 0.583056. The coefficient sign instability is consistent with
the stacker's weak generalization.

## Provenance and resources

All 15 outer folds and 45 inner folds completed exactly once with zero strengthened-group
overlap and both labels. Every inner row received exactly one genuine OOF prediction from each
base model; outer-validation rows used for inner fitting or selection were zero. Each sparse
vectorizer, rare-token map, numeric scaler, Candidate R scaler, base classifier, V selection,
and W stacker used only the applicable training partition. Feature configuration fingerprint:
`08e357e08061b5347e402e44657c4e7ace3e9667bcdc5a9cc8e340bf78290095`.

Wall time was 134.698 seconds. Retrieval used four CPU workers and one corpus index; nested outer
evaluation used three spawn workers with one BLAS thread each. mDeBERTa selected synthetic batch
64, inference took 8.352 seconds, and peak allocated VRAM was 2,530,798,080 bytes (19.65% of
12,878,086,144 bytes). Peak main-process working set was 3,509,911,552 bytes and maximum recorded
system-memory load was 47%.

No competition-test row, leaderboard-derived fitting signal, public labeled data, extra weight,
extra threshold, submission, raw text, or row-level probability was used or persisted.
