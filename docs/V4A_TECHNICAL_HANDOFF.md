# V4-A Technical Handoff

## 1. Executive summary

Version 4-A is a retrieval-assisted lexical classifier for Bengali hallucination
detection. It combines the official 299 labeled examples with an unlabeled
Bengali Wikipedia index. When an example has no supplied context, the system
retrieves a Wikipedia passage from the prompt and uses that passage as provisional
evidence. It then converts the context-response relationship into eight explicit
features and predicts hallucinated (`0`) or faithful (`1`) with class-balanced
logistic regression.

V4-A scored **0.685** on the Kaggle public leaderboard, compared with **0.525**
for V3, an absolute improvement of **0.160**. Its duplicate-aware grouped
validation macro F1 was **0.6872701508**, close to the leaderboard result. The
main practical improvement was not a larger classifier: it was giving
no-context examples useful Bengali evidence and measuring direct lexical and
numeric support between that evidence and the response.

V4-A is the current best project baseline, not a final winning system. It has
limited semantic reasoning, uses only one retrieved article, and was evaluated
on a very small labeled set.

## 2. V3 versus V4-A comparison

| Area | V3 | V4-A |
|---|---|---|
| Core model | Fine-tuned BanglaBERT sequence classifier | Character TF-IDF retrieval plus logistic regression |
| External knowledge | None | 62,150 usable Bengali Wikipedia articles |
| Labeled training data | 299 official rows | 299 official rows |
| Primary input signal | Learned semantics from prompt, context, and response | Explicit support between effective context and response |
| Missing-context handling | Model receives the available text | Retrieve Wikipedia evidence when confidence is sufficient |
| Validation protection | Duplicate-aware grouped folds | Nested 5x3 duplicate-aware grouped validation |
| Deployed decision threshold | Label-1 probability 0.54 | Label-1 probability 0.50; retrieval cutoff 0.25 |
| Public Kaggle score | 0.525 | **0.685** |

The **0.160** leaderboard improvement is consistent with V4-A introducing an
evidence source and task-specific consistency features. It does not prove that
lexical retrieval will beat a stronger semantic model in every setting. V3 and
V4-A also have different inductive biases, which makes their probabilities
useful candidates for a future ensemble.

## 3. Context-present and no-context routing

V4-A creates one *effective context* for every row:

1. The supplied competition context is cleaned using the frozen null-value
   rules.
2. If that context is non-empty, it is always preserved. Wikipedia does not
   replace or augment it.
3. If context is missing, the prompt's highest-scoring Wikipedia result is
   accepted only when its character TF-IDF similarity is at least **0.25**.
4. If the retrieval score is below 0.25, effective context remains empty.

This separation matters. Context-present examples use the evidence supplied by
the task, while no-context examples can gain external evidence without allowing
a weak retrieval result to overwrite known context.

## 4. Bengali Wikipedia indexing and retrieval

The successful run discovered **301** WikiExtractor chunks in the mounted
Bengali Wikipedia dataset. Every selected `wiki_*` chunk was required to parse
as UTF-8 JSON objects. Articles were deduplicated by URL, empty or very short
entries were excluded, and **62,150 usable articles** remained.

For each article, the index stores a search document formed from its title and
the first 800 characters of its body. Wikipedia text is used only as an
unlabeled evidence corpus; it does not add training labels. The retrieval query
is the example's prompt. The single article with the highest cosine similarity
is returned with its similarity score and evidence snippet.

The runtime records the discovered chunk count and aggregate bytes so that a
changed Kaggle mount is visible. It also rejects malformed chunks and ambiguous,
non-identical corpus layouts.

## 5. Character TF-IDF retrieval method

The Wikipedia search index uses scikit-learn `TfidfVectorizer` with:

- analyzer: `char_wb`;
- character n-grams: 2 through 4;
- maximum vocabulary: 50,000 features; and
- similarity: cosine similarity between each prompt vector and article vectors.

Character n-grams are a practical fit for Bengali because they retain useful
surface similarity despite token-boundary, inflection, and minor spelling
variation. They also avoid dependence on a downloaded tokenizer or embedding
model, making CPU-only, internet-disabled Kaggle execution reproducible. This
is lexical retrieval, however: a high score means similar strings and topics,
not necessarily logically correct evidence.

## 6. Retrieval threshold and 800-character evidence snippet

The deployed retrieval cutoff is **0.25**. It was frozen for the final V4-A run
and was not selected from leaderboard feedback. During honest validation, the
retrieval cutoff is selected independently inside each outer fold's three-fold
grouped inner validation from candidates 0.05 through 0.40. The untouched outer
fold is then evaluated with that selected cutoff.

An accepted result contributes only the first **800 characters** of the article
body as evidence. This keeps feature construction bounded and tends to capture
the introductory summary of a Wikipedia article. The tradeoff is that a fact
located later in the article is invisible to the classifier.

The probability decision threshold is separate from the retrieval threshold:
the logistic regression predicts label `1` at probability **0.50** or above.

## 7. Exact eight classifier features

The classifier receives these features in this exact order:

1. `ctx_present` — 1 when effective context is non-empty, otherwise 0.
2. `word_overlap_ratio` — unique response word tokens also found in context,
   divided by the number of unique response word tokens.
3. `numbers_supported` — 1 when every number in the response is present in the
   effective context; Bengali digits are normalized to ASCII for comparison.
4. `number_overlap_ratio` — the fraction of response numbers found in context.
5. `has_numbers_in_response` — 1 when the response contains at least one number.
6. `is_substring` — 1 when the complete stripped response occurs exactly inside
   the effective context.
7. `response_len_chars` — response length in characters.
8. `context_len_chars` — effective-context length in characters.

The feature matrix contains no prompt or response text. Features are standardized
using parameters learned from the training fold before the classifier is fit.

## 8. Class-balanced logistic regression

V4-A uses `LogisticRegression(max_iter=1000, class_weight="balanced",
random_state=42)` after `StandardScaler`. Class balancing weights each class
inversely to its training frequency so the decision is not driven solely by the
more common label. This is important because label `0` means hallucinated and
label `1` means faithful, and a deceptively strong class-specific score can be
produced by collapsing toward one label.

For every validation fold, scaling and classifier fitting use only that fold's
training rows. After validation choices are frozen, the final scaler and model
are trained on all 299 official labeled rows. The code locates the label-1
probability column from `model.classes_` rather than assuming a column order.

## 9. Duplicate-aware grouped validation

Ordinary random folds can leak near-duplicate examples across training and
validation. V4-A therefore builds groups that join:

- exact normalized prompt/context/response duplicates; and
- all response variants that share the same normalized prompt and context.

Conflicting labels inside an exact duplicate group are rejected. Five outer
`StratifiedGroupKFold` folds keep every group wholly on one side of the split,
require both labels in every validation fold, and cover every labeled row once.

Within each outer-training partition, a separate three-fold grouped split
selects the retrieval cutoff. Only then is the selected cutoff applied to the
untouched outer-validation fold. The resulting OOF probabilities are therefore
the honest estimate to use for comparison and blending. The implementation also
reports a same-OOF original-method estimate, but explicitly labels that estimate
optimistic; it should not replace the nested grouped result.

## 10. Validation and Kaggle results

| Result | Value |
|---|---:|
| Grouped accuracy | **0.6889632107** |
| Grouped macro F1 | **0.6872701508** |
| Grouped label-0 F1 | **0.7102803738** |
| Grouped label-1 F1 | **0.6642599278** |
| Kaggle public score | **0.685** |
| Usable Wikipedia articles | 62,150 |
| Wikipedia chunks | 301 |
| Test predictions, label 0 | 1,664 |
| Test predictions, label 1 | 852 |

The test output contains 2,516 predictions: about 66.1% label `0` and 33.9%
label `1`. Neither class exceeded the project's 90% collapse-warning threshold.
Submission IDs, schema, row count, label domain, and row alignment were all
validated before the file was written.

A teammate's reported **69%** result is comparable only if it uses the same
metric, the same split, and the same duplicate-aware validation procedure. A
plain accuracy value, a different random split, or a split that allows related
examples across folds is not an apples-to-apples comparison with V4-A's grouped
macro F1.

## 11. Why the approach improved performance

The most credible contributors are:

- **Evidence for missing context.** No-context rows can now be assessed against
  a relevant Bengali reference passage instead of relying only on learned
  correlations in the prompt and response.
- **Direct task signals.** Word overlap, number support, full-response substring
  matching, and context/response lengths encode observable consistency rather
  than asking a small fine-tuning set to teach those rules implicitly.
- **Bengali-native lexical retrieval.** Character n-grams provide robust lookup
  across Bengali surface variation without a large retrieval model.
- **Low variance on 299 labels.** A balanced linear classifier over eight
  structured features is much easier to estimate reliably than a high-capacity
  neural classifier on such a small dataset.
- **Leakage-resistant model selection.** Nested grouped validation gave a more
  realistic basis for the retrieval decision and produced an estimate close to
  the public score.

These are evidence-based explanations from the implementation and observed
results, not a causal decomposition. One leaderboard score cannot isolate the
gain from each component.

## 12. Known weaknesses and failure cases

- Retrieval uses only the prompt, returns one article, and has no semantic
  reranker. A topically similar but factually wrong article can win.
- The relevant fact may be absent from Wikipedia or located after the
  800-character snippet.
- A global 0.25 cutoff is not calibrated per topic or query difficulty.
- Context-present rows are never supplemented by Wikipedia, even when their
  supplied context is incomplete.
- Word overlap uses sets, losing order and repetition. Exact substring matching
  is brittle to paraphrase, punctuation, and normalization differences.
- Numeric support catches copied or conflicting numbers but cannot determine
  whether a supported number answers the prompt correctly.
- The linear classifier cannot model rich interactions or semantic entailment.
- Validation has only 299 labeled rows, so fold metrics and model-selection
  decisions remain uncertain.
- The Wikipedia source can have coverage, date, spelling, and entity-resolution
  gaps. Runtime corpus shape is validated and recorded, but not pinned by a
  strict per-file historical manifest.
- The public score is a single leaderboard observation and may not represent
  private-test performance.

## 13. Outputs to combine with the next GPU model

The useful preserved outputs are:

- `v4a_oof_probabilities.csv` — honest grouped OOF label-1 probabilities, with
  source row index and validation fold for all 299 labeled rows;
- `v4a_test_probabilities.csv` — label-1 probabilities aligned by official test
  ID for all 2,516 rows; and
- `v4a_run_summary.json` — aggregate configuration, validation metrics, group
  audit, corpus counts, and runtime metadata.

Use the OOF and test probabilities for blending. Do not optimize a blend from
`submission.csv`: its hard labels discard confidence, and leaderboard-based
weight selection would be probing. The runtime files are preserved under the
Git-ignored `artifacts/kaggle/v4a_0685/` directory; their checksums are recorded
in `docs/EXPERIMENT_ARTIFACT_INVENTORY.md`.

## 14. Practical recommendations for a teammate building another model

1. Generate the new model's OOF probabilities on the same 299 official rows,
   using the same duplicate groups and preferably the same outer fold assignment.
2. Keep every probability genuinely out of fold. Do not blend V4-A OOF values
   with in-sample probabilities from a final-fit GPU model.
3. Emit probability for label `1` explicitly and preserve full precision.
4. Align test predictions by validated `id`, never by an assumed row order.
5. Compare V4-A, the new model, and their blend with ordinary two-class macro F1
   under the same grouped procedure. Report label-specific F1 and accuracy too.
6. Select blend weights and any probability threshold only inside training/OOF
   validation. Do not use public-leaderboard feedback to tune them.
7. Favor a complementary semantic model: multilingual or Bengali entailment
   reasoning can address paraphrases and subtle contradictions that V4-A's
   lexical features miss.
8. Preserve safe aggregate diagnostics, fold assignments, configuration, and
   hashes so another teammate can reproduce the comparison without accessing
   competition test text.

The strongest next step is a leakage-safe probability ensemble: let V4-A
contribute explicit evidence-consistency signals while the GPU model contributes
semantic understanding.
