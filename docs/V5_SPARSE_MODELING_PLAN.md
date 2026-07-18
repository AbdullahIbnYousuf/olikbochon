# V5 Context-Absent Sparse Modeling Plan

## Scope and preparation status

Experiment identity: `v5_null_sparse_baseline`.

This is one official-labeled-sample-only, CPU sparse-text experiment. Only the 169 corrected
context-absent rows may fit the five sparse classifiers. Every context-present validation
decision remains the frozen Candidate A exact normalized substring rule; present rows never fit
a sparse vocabulary or classifier. Competition-test data, public 5K/20K data, neural models,
downloads, and leaderboard feedback are excluded.

Preparation was frozen before evaluation. The single approved experiment has now completed;
numeric results and gate verdicts are recorded in `V5_SPARSE_EXPERIMENT_LOG.md`.

## Frozen normalization and text views

All fields reuse `normalize_v5`: NFC, zero-width removal, numeral equivalence, fixed punctuation
equivalence, Latin-only lowercasing, and whitespace normalization. Bengali words, numerals,
negation, uncertainty terms, and punctuation are retained. No stemming or punctuation-stripped
vectorized channel is declared.

View 1 is exactly:

```text
[QUESTION] <normalized prompt>
[ANSWER] <normalized response>
```

The markers are ordinary text. View 2 independently vectorizes normalized prompt and response.
View 3 is the View-1 character+word union followed by all 21 already-frozen context-absent
numeric features. Those features include prompt-response overlap, response length, rare-token
state, and specificity.

## Exact vectorizer and classifier configuration

Every `TfidfVectorizer` uses `lowercase=False`, `sublinear_tf=True`, `min_df=2`, and `norm="l2"`.

| Channel | Analyzer | N-grams | Maximum features |
|---|---|---:|---:|
| Combined character | `char_wb` | 3–5 | 30,000 |
| Combined word | `word` | 1–2 | 15,000 |
| Separate prompt | `word` | 1–2 | 8,000 |
| Separate response | `char_wb` | 3–5 | 20,000 |

Every candidate uses exactly:

```text
LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=3000,
    solver="liblinear",
    random_state=42,
)
```

Candidate I uses `StandardScaler` fitted on its current null-route training fold for the 21
numeric features. Sparse and scaled-numeric matrices are horizontally concatenated. No C
retuning or alternate classifier is present.

## Frozen candidates

- Candidate E, `candidate_e_char_tfidf`: View 1 with combined character TF-IDF.
- Candidate F, `candidate_f_word_tfidf`: View 1 with combined word TF-IDF.
- Candidate G, `candidate_g_char_word_union`: independent View-1 character and word
  vectorizers, horizontally concatenated.
- Candidate H, `candidate_h_separate_fields`: View 2 with prompt word and response character
  vectorizers, horizontally concatenated.
- Candidate I, `candidate_i_sparse_lexical_union`: Candidate G sparse channels plus the frozen
  21 context-absent numeric features.

No SVM, Naive Bayes, tree model, boosting, neural embedding, ensemble, additional channel, or
hyperparameter search is included.

## Fit isolation and leakage proof

For every seed/fold/candidate, the runner first selects only the context-absent training indices,
constructs a fresh `SparseNullModel`, calls each vectorizer's `fit_transform` only on that
training frame, and calls `transform` on the null validation frame afterward. A model instance
cannot transform before fitting. Candidate I similarly fits a fresh `V5FeatureExtractor` and
`StandardScaler` on the training frame before validation transformation.

Focused synthetic tests prove that a token occurring only in validation never enters the fitted
word vocabulary or Candidate-I response-frequency map. They also verify identical probabilities
from repeated fits, finite sparse/numeric matrices for all five candidates, and training-row-only
fit audit metadata. Vocabulary terms are never serialized; only per-channel vocabulary sizes and
training-row counts may enter artifacts.

The runner accepts no input path, data role, model path, or individual candidate override. It
loads the authenticated official sample through the existing fixed-path hash-checking loader.
Its only path argument must resolve exactly to the ignored new output directory.

## Grouped validation feasibility

The exact V5 group graph is reused: normalized exact rows, prompt identity, present-context
identity, inherited family groups, and inherited near-duplicate groups. Context-absent sentinel
values are not collapsed into one group.

The read-only feasibility audit found:

- 299 authenticated official rows and 291 groups;
- largest group: 2;
- seeds: 17, 29, 43;
- five folds per seed, 15 total;
- zero train/validation group overlap;
- both labels in every route-specific train and validation subset;
- null training rows: 129–140; label 0: 67–74; label 1: 59–66;
- null validation rows: 29–40; label 0: 15–22; label 1: 14–21.

There is no naive-random fallback.

## Routed evaluation and metrics

Within each validation fold, context-present rows are predicted only by Candidate A. Each sparse
candidate produces probabilities only for context-absent rows. Threshold 0.50 is primary. The
secondary 0.20–0.80 grid in increments of 0.02 is selected on null-route OOF probabilities only;
Candidate A is never tuned. The selected null threshold is then combined with the fixed present
decisions for routed metrics.

Each null candidate reports macro F1, both label F1 values, accuracy, confusion matrix, predicted
counts, Brier score, probability mean/SD, selected threshold, collapse guard, and seed
mean/SD/min/max. Routed reporting adds route macro F1 values and seed distributions. Artifacts
contain only frozen config, fold/seed/aggregate numeric metrics, training-row counts, and
vocabulary sizes—never raw text, vocabulary terms, row probabilities, predictions, or feature
matrices.

## Frozen acceptance rules

Null route:

- weak below 0.55;
- promising at least 0.58;
- strong at least 0.62;
- high-value at least 0.68.

Routed:

- must exceed the current 0.683111 champion to replace it;
- meaningful gain is at least +0.01;
- strong overall is at least 0.70;
- high-value overall is at least 0.74.

If no sparse candidate beats the null-route baseline 0.541763, sparse lexical research stops and
a neural/retrieval route should be considered. These gates are immutable after evaluation.

## Executed command

The experiment was executed exactly once with:

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"

.\.venv\Scripts\python.exe -m olikbochon.v5_sparse_runner `
  --mode null-sparse-baseline `
  --candidate-set frozen `
  --seeds 17 29 43 `
  --folds 5 `
  --output-dir "F:\datathon\olikbochon\artifacts\v5\null_sparse_baseline"
```

Observed CPU wall time was approximately 20.2 seconds. Aggregate-only storage was 611,638 bytes.
No GPU, model download, checkpoint, or model artifact was used.
