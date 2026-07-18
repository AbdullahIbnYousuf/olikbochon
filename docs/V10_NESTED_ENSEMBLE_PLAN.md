# V10 Nested Lexical-Semantic Ensemble Plan

Status before execution: **frozen**.

V10 evaluates exactly three probability-level ensembles of frozen Candidate I and Candidate R
on the authenticated 169-row context-absent route. The 130 context-present rows retain the exact
substring rule. Candidate I is the unchanged V5 `candidate_i_sparse_lexical_union`. Candidate R
is the unchanged V9 14-feature standardized logistic model using seven frozen mDeBERTa semantic
aggregates and seven Candidate N retrieval aggregates. mDeBERTa revision is
`b5113eb38ab63efdd7f280f8c144ea8b13f978ce`; decision threshold is 0.50.

## Candidates

- U: fixed `0.50 * P(I) + 0.50 * P(R)`.
- V: select the Candidate-I weight from exactly `0.00, 0.25, 0.50, 0.75, 1.00` using current
  outer-training inner OOF macro F1. Deterministic ties use label-0 F1, proximity to 0.50, then
  lower Candidate-I weight.
- W: a standardized balanced liblinear logistic stacker with exactly the two inner OOF
  probabilities as inputs and the frozen `C=1.0`, `max_iter=3000`, `random_state=42` settings.

Every outer fold independently creates three group-safe inner folds from its training rows.
Candidate I and Candidate R are refitted on each inner-training partition and predict only its
inner-validation partition. These complete inner OOF probabilities select V and fit W. Both base
models are then refitted on the complete outer-training partition and predict untouched outer
validation. No outer-validation row affects a vectorizer, rare-token map, scaler, base classifier,
blend choice, or stacker.

Outer validation uses seeds 17, 29, and 43 with five strengthened grouped folds and three spawn
workers. Eligibility requires routed macro F1 at least 0.697051, paired improvement over Candidate
I in at least two seeds, seed standard deviation no greater than 0.04, genuine nested provenance,
and no predicted class over 90%.

Exact command:

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

.\.venv\Scripts\python.exe -m olikbochon.v10_runner `
  --mode nested-lexical-semantic-ensemble `
  --corpus-path "F:\datathon\olikbochon\data\retrieval\bnwiki" `
  --primary-model-path "F:\datathon\olikbochon\data\models\v9_nli\mdeberta_xnli@b5113eb38ab63efdd7f280f8c144ea8b13f978ce" `
  --seeds 17 29 43 `
  --folds 5 `
  --cpu-workers 4 `
  --outer-workers 3 `
  --output-dir "F:\datathon\olikbochon\artifacts\v10\nested_lexical_semantic_ensemble"
```

The experiment does not access competition-test rows, use leaderboard information for fitting,
tune another weight or threshold, add features, produce a submission, or persist raw text and
row-level probabilities.
