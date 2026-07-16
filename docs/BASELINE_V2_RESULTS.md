# Baseline Version 2 Results

## Scope

Version 2 keeps the exact Version 1 TF-IDF and balanced logistic-regression configuration. It changes only the amount and role of approved labeled data. Development fitting used `bangla_hallucination_5k_train.json`, threshold selection used `bangla_hallucination_5k_validation.json`, and the official 299-row `dataset samples.json` remained untouched until independent evaluation. The 5k `bangla_hallucination_5k_contrastive.json` was audit-only. No model or threshold decision used competition test data or leaderboard feedback.

The exact and near-duplicate gates passed with zero removals. Counts were therefore:

| Role | Rows |
|---|---:|
| Development training | 4,000 |
| Public threshold validation | 1,000 |
| Independent official evaluation | 299 |
| Final combined unique refit | 5,299 |

Full filenames, hashes, distributions, fingerprint definitions, and similarity results are in `docs/V2_DATA_AUDIT.md`.

## Locked model

- Marked text: `__PROMPT__`, `__CONTEXT_PRESENT__`, `__CONTEXT__`, `__RESPONSE__`.
- Preprocessing: NFC, normalized missing context, whitespace collapse, preserved Bengali/punctuation/numbers.
- Word TF-IDF: word 1–2 grams, 20,000 cap, `min_df=1`, sublinear TF, case preserved, `float32`, L2.
- Character TF-IDF: `char_wb` 3–5 grams, 30,000 cap, otherwise matching fixed settings.
- Classifier: balanced `LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=42)`.
- Threshold grid: 0.20–0.80 by 0.01; macro F1, then class-0 F1, proximity to 0.50, then lower threshold.

## Public validation

The selected threshold was **0.50**. Because the public 1k split both selected and evaluated it, the selected score is explicitly a **Public-validation threshold-tuning estimate**, not an independent estimate. In this run, it equals the fixed-threshold result.

| Result | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix | Predicted 0 / 1 |
|---|---:|---:|---:|---:|---|---:|
| Threshold 0.50 | 0.400000 | 0.336842 | 0.368421 | 0.370000 | `[[210, 290], [340, 160]]` | 550 / 450 |
| Threshold-tuning estimate, selected 0.50 | 0.400000 | 0.336842 | 0.368421 | 0.370000 | `[[210, 290], [340, 160]]` | 550 / 450 |

Neither result exceeded the 90% predicted-class warning limit.

## Independent official-sample evaluation

The public-trained model was applied once to the untouched official sample at fixed 0.50 and the already frozen selected threshold. The threshold was not changed after seeing these results. Because the selected threshold is 0.50, both rows are identical.

| Result | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix | Predicted 0 / 1 |
|---|---:|---:|---:|---:|---|---:|
| Threshold 0.50 | 0.569106 | 0.305677 | 0.437391 | 0.468227 | `[[105, 31], [128, 35]]` | 233 / 66 |
| Selected threshold 0.50 | 0.569106 | 0.305677 | 0.437391 | 0.468227 | `[[105, 31], [128, 35]]` | 233 / 66 |

### Official context regimes

| Regime | Count | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Context absent | 169 | 0.615385 | 0.134615 | 0.375000 | 0.467456 | `[[72, 17], [73, 7]]` |
| Context present | 130 | 0.488889 | 0.448000 | 0.468444 | 0.469231 | `[[33, 14], [55, 28]]` |

No official prediction set exceeded the 90% collapse-warning limit.

## Comparison with Version 1

Version 1's official-sample fixed-0.50 OOF macro F1 was 0.520686; its full-OOF macro-selected tuning estimate was 0.524823. Version 2's independent official-sample macro F1 is 0.437391. This is not a perfectly identical comparison: Version 1 used official-sample cross-validation, whereas Version 2 fits on separate public data and treats the official sample as a domain holdout. The lower V2 result is evidence of a public-to-official distribution or task mismatch, not a reason to tune on the official sample or public leaderboard.

## Runtime and environment

- Local Python: 3.14.4
- NumPy: 2.5.1
- pandas: 3.0.3
- scikit-learn: 1.9.0
- Self-contained local notebook-source run: 9.52 seconds on CPU
- Local competition-test inference: deliberately skipped

## Final-refit calibration limitation

The threshold remains fixed at 0.50 after selection. The final Kaggle model is refitted on all 5,299 unique allowed rows, including the former validation and official rows. Adding these rows can shift probability calibration, so the frozen threshold may be slightly less optimal for the final refitted model. Version 2 accepts this limitation and introduces no second tuning stage. Kaggle scores must not be used for repeated threshold probing.

## Safety confirmations

- No competition test text was inspected or loaded locally.
- Every bundled test copy is excluded from labeled-data roles.
- No external API, network request, transformer, model download, or GPU code was used.
- No raw examples, row-level probabilities, or predictions are committed.
- The public aggregate is audit-only and is never added alongside its 4k/1k partitions for fitting.
