# Locked current-champion inference plan

Status: **prepared and not executed**. The competition test CSV was not opened, parsed,
hashed, or displayed during this preparation. Final inference and artifact creation require
separate approval.

## Frozen champion

- training data: authenticated 299-row official labeled sample only;
- routing: audited absent-context policy covering Python/JSON null, NaN, empty or
  whitespace-only values, and normalized `[NULL]` variants;
- present route: Candidate A, label 1 only when the normalized response is an exact
  substring of normalized context;
- absent route: Candidate I fitted on all 169 official absent-context rows;
- Candidate I channels: combined `char_wb` TF-IDF 3–5 grams with 30,000 maximum features,
  combined word TF-IDF 1–2 grams with 15,000 maximum features, and all 21 frozen numeric
  absent-route features;
- both TF-IDF channels: `lowercase=False`, `sublinear_tf=True`, `min_df=2`, `norm=l2`;
- numeric features: `StandardScaler` and rare-response-token map fitted only on the 169
  absent official training rows;
- classifier: balanced liblinear logistic regression, C=1.0, 3,000 maximum iterations,
  random state 42;
- decision threshold: exactly 0.50; no threshold search or override exists.

V4-A probabilities, V7 retrieval, Bengali Wikipedia, BanglaBERT, public labeled data,
external APIs, manual review, and leaderboard feedback are excluded.

## Locked paths and authentication

The CLI accepts only these absolute paths:

- train: `F:\datathon\olikbochon\data\competition\dataset samples.json`;
- test: `F:\datathon\olikbochon\data\competition\test set.csv`;
- output: `F:\datathon\olikbochon\artifacts\submissions\current_champion_0692051`.

The expected official test authentication is frozen before parsing:

- filename: `test set.csv`;
- byte size: 2,329,947;
- SHA-256: `db75049956c6fa00e4d9c476716ee34bc4cc17a737f52ada06d2c0f80d567b81`;
- expected rows: 2,516;
- exact columns: `id`, `context`, `prompt_bn`, `response_bn`;
- IDs: required, nonmissing, unique, and preserved in original order.

The official training SHA-256 remains
`f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28`.
Public/retrieval/model paths, alternate output locations, candidate overrides, and threshold
overrides are rejected. Existing output is a fatal error; no overwrite mode exists.

## Output contract

The ignored output directory is created only after authentication, fitting, inference, and
validation succeed. It contains:

1. `submission.csv`: exactly `id,label`, 2,516 ordered unique IDs, integer labels 0/1,
   no index or missing values;
2. `champion_test_probabilities.csv`: `id`, `route`, `label_1_probability`,
   `deterministic_rule_score`, and `predicted_label`;
3. `run_summary.json`: aggregate configuration, counts, dimensions, runtime, memory, and
   hashes only;
4. `artifact_checksums.json`: SHA-256 values for the submission, probability artifact,
   and run summary.

For absent rows, `label_1_probability` contains Candidate I output and
`deterministic_rule_score` is empty. For present rows, `label_1_probability` is empty and
`deterministic_rule_score` is the Candidate A hard decision. Hard rule decisions are never
represented as calibrated probabilities. No prompt, context, response, vocabulary term,
row-level feature vector, model, or fitted vocabulary is persisted.

## Proposed command — not yet executed

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"

.\.venv\Scripts\python.exe -m olikbochon.champion_submission_runner `
  --mode champion-submission `
  --train-path "F:\datathon\olikbochon\data\competition\dataset samples.json" `
  --test-path "F:\datathon\olikbochon\data\competition\test set.csv" `
  --output-dir "F:\datathon\olikbochon\artifacts\submissions\current_champion_0692051"
```

Expected wall time is under one minute on the verified GPU PC, although the frozen model is
CPU-only. Expected generated storage is below 1 MB and expected peak process memory is below
1 GB. These are capacity estimates, not observed test-inference measurements.

## Dry-verification coverage

Synthetic tests prove exact 299/130/169 training counts, Candidate I configuration, fixed
threshold, both Candidate A rule outcomes, strict test ID/order handling, test-to-fit
isolation, valid binary submission labels, honest route-specific score fields, no raw text in
outputs, deterministic repeated inference, ignored output, and overwrite/path rejection.
They do not open the real test CSV.
