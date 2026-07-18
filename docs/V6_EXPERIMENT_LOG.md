# V6 official-only null-route neural experiment

Status: **completed**. The one authorized frozen experiment ran 45 fits: Candidates J,
K, and L across seeds 17, 29, and 43 with five strengthened grouped folds per seed. No
competition-test rows, public data, V4-A probabilities, retrieval corpus, predictions, or
submissions were used.

## Execution and safety audit

The exact command was:

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

- authenticated official rows: 299;
- corrected routes: 130 context-present and 169 context-absent;
- null-route labels: 89 label 0 and 80 label 1;
- neural fitting scope: context-absent only;
- group overlap: zero in all 45 fits;
- both labels: present in every train and validation route split;
- input serialization: question and answer only; context and `[NULL]` absent;
- maximum original null-route length: 59 tokens; truncated rows: zero;
- CUDA fp16 remained available;
- non-finite losses/probabilities: zero;
- selected checkpoints reload-verified exactly: 45 of 45;
- persisted raw text or row-level probabilities: none;
- artifacts remained under ignored `artifacts/v6/null_neural_baseline/`.

The shared schedule was six epochs, batch size 8, gradient accumulation 2, AdamW weight
decay 0.01, 10% linear warmup, fold-training-only balanced class weights, gradient norm
1.0, and zero data-loader workers. Checkpoints were ranked by validation macro F1 at
threshold 0.50, then lower validation loss, then earlier epoch.

## Null-route aggregate results

The selected threshold is one global frozen-grid threshold per candidate. Validation loss
is the mean selected-fold weighted validation loss across 15 fits.

| Candidate | Threshold | Macro F1 | Label-0 F1 | Label-1 F1 | Accuracy | Confusion matrix | Predicted `[0,1]` | Validation loss | Brier | Probability mean/std |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| J at 0.50 | 0.50 | 0.382171 | 0.692913 | 0.071429 | 0.538462 | `[[88,1],[77,3]]` | `[165,4]` | 0.692377 | 0.248374 | 0.478962 / 0.011654 |
| J selected | 0.48 | 0.520105 | 0.537143 | 0.503067 | 0.520710 | `[[47,42],[39,41]]` | `[86,83]` | 0.692377 | 0.248374 | 0.478962 / 0.011654 |
| K at 0.50/selected | 0.50 | 0.502802 | 0.511628 | 0.493976 | 0.502959 | `[[44,45],[39,41]]` | `[83,86]` | 0.693125 | 0.249952 | 0.500412 / 0.005999 |
| L at 0.50/selected | 0.50 | 0.510845 | 0.441379 | 0.580311 | 0.520710 | `[[32,57],[24,56]]` | `[56,113]` | 0.693453 | 0.250260 | 0.504342 / 0.010421 |

Selected-threshold null-route seed macro-F1 distributions:

| Candidate | Mean | Std. | Minimum | Maximum | Collapse guard | Versus 0.541763 |
|---|---:|---:|---:|---:|---|---:|
| J | 0.485924 | 0.016095 | 0.466212 | 0.505636 | Pass | -0.021658 |
| K | 0.542252 | 0.025141 | 0.518281 | 0.576979 | Pass | -0.038961 aggregate |
| L | 0.518237 | 0.015251 | 0.498554 | 0.535714 | Pass | -0.030918 |

The comparison column uses each candidate's aggregate selected-threshold macro F1, not the
mean of the three independently scored seeds. J and K fail the primary 0.58 pass gate; L
is diagnostic only and also falls below 0.55. All three are classified as **failed** null
models. None beats the handcrafted null-route champion 0.541763.

## Routed aggregate results

Every routed result combines the frozen substring rule on the 130 context-present rows
with the candidate's null-route prediction on the 169 context-absent rows. The fixed
present-route macro F1 is 0.900026 for every candidate.

| Candidate | Threshold | Macro F1 | Label-0 F1 | Label-1 F1 | Accuracy | Confusion matrix | Predicted `[0,1]` | Present F1 | Absent F1 |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| J at 0.50 | 0.50 | 0.690690 | 0.741379 | 0.640000 | 0.698997 | `[[129,7],[83,80]]` | `[212,87]` | 0.900026 | 0.382171 |
| J selected | 0.48 | 0.685800 | 0.654275 | 0.717325 | 0.688963 | `[[88,48],[45,118]]` | `[133,166]` | 0.900026 | 0.520105 |
| K at 0.50/selected | 0.50 | 0.674971 | 0.639098 | 0.710843 | 0.678930 | `[[85,51],[45,118]]` | `[130,169]` | 0.900026 | 0.502802 |
| L at 0.50/selected | 0.50 | 0.675913 | 0.610879 | 0.740947 | 0.688963 | `[[73,63],[30,133]]` | `[103,196]` | 0.900026 | 0.510845 |

Selected-threshold routed seed macro-F1 distributions:

| Candidate | Mean | Std. | Minimum | Maximum | Versus 0.692051 |
|---|---:|---:|---:|---:|---:|
| J | 0.674107 | 0.010741 | 0.661608 | 0.687832 | -0.006251 aggregate |
| K | 0.696849 | 0.017329 | 0.680735 | 0.720897 | -0.017080 aggregate |
| L | 0.682745 | 0.007778 | 0.672185 | 0.690690 | -0.016138 aggregate |

All candidates pass the seed-standard-deviation limit of 0.04 and the class-collapse
guard. None beats the routed champion 0.692051; none reaches strong overall status at
0.70. Candidate J is the selected V6 candidate because it has the highest aggregate
selected-threshold routed macro F1 and is also the lowest-capacity model. It is not a new
overall champion.

## Training, VRAM, and checkpoint authentication

| Candidate | Trainable parameters | Selected epochs distribution | Fold runtime | Peak allocated VRAM | Retained checkpoint bytes | Representative SHA-256 |
|---|---:|---|---:|---:|---:|---|
| J | 1,538 | `1:5, 2:3, 4:3, 5:1, 6:3` | 45.300 s | 478,901,248 B | 6,478 | `40215dc0af21bba0dce7690b223f2f1500d43d74ef3c0c4f24f167f2a28a06f2` |
| K | 14,767,874 | `1:1, 2:2, 3:3, 4:3, 5:3, 6:3` | 52.977 s | 728,341,504 B | 59,075,809 | `ac39a3596c4abf21a3f4e7af2d4004309044dd976a091d2ffb74ac60852d915b` |
| L | 110,618,882 | `1:2, 2:3, 3:2, 4:4, 5:1, 6:3` | 107.017 s | 2,250,871,808 B | 442,499,784 | `1c5d5766f5376f71efd2702be5912deda4b275688c058cc0e5f8321df3cf6369` |

Total fold runtime was 205.293 seconds; command wall time was 213.2 seconds. Maximum
allocated VRAM was 2,250,871,808 bytes (Candidate L). Selected checkpoints written across
all folds totaled 7,523,731,065 transient bytes. Compact retention preserved exactly
three seed-17/fold-1 checkpoints totaling 501,582,071 bytes. The other 42 checkpoints were
deleted only after exact reload verification. The immutable base snapshot was preserved.

## Verdict

- best V6 null model: Candidate J at threshold 0.48, macro F1 0.520105;
- best V6 routed model: Candidate J at threshold 0.48, macro F1 0.685800;
- null champion remains handcrafted lexical logistic at 0.541763;
- routed champion remains substring plus Sparse Candidate I at 0.692051;
- neural seed instability and class collapse were not the failure modes;
- probabilities remained concentrated near 0.50 and validation losses near 0.693,
  indicating weak null-route discrimination.

Under the frozen decision rules, all three V6 neural candidates are rejected. The
predeclared direction after no J/K pass is retrieval/NLI; because the separately audited
V4-A retrieval null route also failed, the evidence favors NLI or stopping further
official-only null-route modeling rather than neural/retrieval blending.
