# Validation Plan

## Objectives

Validation must estimate generalization to unseen official items, support threshold selection without touching competition test text, reveal context-regime failures, and remain cheap enough for a beginner team to repeat.

The official metric wording is ambiguous. Until clarified, every run must report both class-specific F1 values and macro F1, with class-0 F1 treated as the provisional decision metric.

## Data admission and deduplication before splitting

1. Begin with an explicit allowlist of labeled JSON files.
2. Exclude every CSV from model development, especially the public file whose hash equals the official test.
3. Include only one copy of the official sample.
4. If public data is approved, include either the 5k aggregate or its 4k+1k partition, never both.
5. Normalize text only for duplicate detection and splitting; preserve original normalized model inputs separately.
6. Form an exact group key from normalized `(context, prompt_bn, response_bn)`.
7. Detect conflicting labels within exact groups and stop for review rather than choosing a label.
8. Build near-duplicate groups using word/character TF-IDF similarity, MinHash, or normalized edit similarity. Perform this on labeled data only. Review only aggregate counts unless a tiny labeled example is genuinely needed.
9. Keep every exact/near-duplicate family wholly within one fold.

Because synthetic contrastive data may share most of a prompt/context while changing one fact, grouping only exact rows is insufficient. At minimum, create secondary groups from normalized prompt plus context and test sensitivity to the grouping rule.

## Split designs

### Version 1: official sample only

- Use 5-fold stratified CV with shuffle and seed 42.
- Stratify on label; where feasible, balance the joint `(label, has_context)` distribution without creating tiny strata.
- Use group-aware assignment if duplicate-family checks find groups.
- The sample is small, so fold variance is part of the result, not noise to hide.
- Keep the locked outer seed 42 for Version 1. Report all five fold metrics and their mean/standard deviation rather than selecting a favorable fold or seed.

### If the public dataset is approved

- Preserve the public source’s intended 4k train/1k validation partition as one diagnostic, but audit near-duplicate leakage across it.
- Prefer training/tuning on approved public data while reserving the official 299-row sample as an untouched domain holdout during model selection. This tests transfer to the authoritative competition distribution.
- After the approach and threshold are frozen, the official sample may be added for the final fit; never report that refit as a new validation result.
- If the public split fails leakage checks, rebuild duplicate-group-aware folds over the 5k aggregate.

### Transformers

Use one fixed duplicate-aware train/validation split for early experiments, then at most 3 folds for the strongest configuration. Do not run expensive multi-seed transformer CV until the simple model and packaging are stable.

## Metrics

For each fold, aggregate OOF predictions, each context regime, and any untouched holdout, calculate:

- **F1 for label 0:** treat hallucinated (`0`) as the positive class.
- **F1 for label 1:** detects whether improving class 0 collapses faithful performance.
- **Macro F1:** arithmetic mean of the two class F1 scores.
- **Confusion matrix:** rows true `[0,1]`, columns predicted `[0,1]`, always with an explicit label order.
- **Accuracy:** diagnostic only, never the sole model-selection criterion.
- **Precision and recall for label 0:** show whether a threshold gain comes from catching more hallucinations or producing excessive false alarms.
- **Context-present and context-absent versions** of class-0 F1, class-1 F1, macro F1, and sample counts.

Why accuracy is insufficient: it weights every correct prediction equally, hides the balance between hallucination precision and recall, and can look strong when a model favors the more common/easier class. F1 directly penalizes a model that misses hallucinations or flags too many faithful responses.

## Threshold selection

For logistic regression, obtain probabilities and locate the class-0 column from `model.classes_`; never assume a column index. Predict class 0 when its probability exceeds threshold `t`.

The locked Version 1 honest procedure is nested threshold evaluation:

1. create outer 5-fold stratified CV with shuffle and seed 42;
2. inside outer fold `i`, create inner 3-fold stratified CV over only the outer-training records with seed `42 + i`;
3. generate inner OOF probabilities and select `t` from 0.20 through 0.80 by 0.01;
4. rank thresholds by class-0 F1, then macro F1, then closeness to 0.50, then lower numerical value;
5. fit on the complete outer-training portion and score the untouched outer-validation portion at that independently selected threshold;
6. aggregate the five outer-validation predictions as the “Nested-CV threshold-selected estimate”; and
7. report each outer threshold plus fold and aggregate metrics.

Separately, generate ordinary full 5-fold OOF probabilities. Report threshold-0.50 metrics as the fixed baseline. Select the final deployment threshold from the full OOF probabilities using the same grid and tie-breaks, save the complete aggregate threshold table, and label its selected-threshold score “Full-OOF threshold-tuning estimate.” This same-OOF selected score is optimistic and is not the primary honest threshold-tuned estimate.

Use the full-OOF-selected threshold for the final all-labeled-data fit and Kaggle inference. Once the organizers clarify the evaluator, change the optimization objective explicitly and version the config.

For a linear SVM, tune on signed decision scores. For ensembles, calibrate/blend on training-fold predictions only and repeat the same outer evaluation.

## Final untouched holdout

- If only the small official sample is usable, the locked nested CV is more informative than permanently sacrificing a large holdout; clearly state that no untouched local holdout exists.
- If the public data is approved, keep the entire official sample untouched through model/feature/threshold selection where feasible.
- Never use the competition test or public leaderboard to choose preprocessing, features, models, ensemble weights, or thresholds.

## Leakage checks per run

- Exact signature intersection across train and validation: must be zero.
- Near-duplicate group intersection: must be zero under the selected grouping rule.
- Source-file overlap: verify no aggregate plus subset duplication.
- Conflicting-label groups: must be zero or explicitly resolved from authoritative documentation.
- Vectorizer/preprocessor fit: training fold only.
- Calibration/threshold fit: training/tuning predictions only.
- Test file hash or path in any training manifest: fatal.
- Target-derived or source-name shortcut features: forbidden unless scientifically justified and stable for held-out data.

## Experiment record

For each run, store only non-sensitive metadata in ignored outputs:

- run ID, Git commit, UTC timestamp, seed, split/group hashes;
- included data filenames and hashes, never raw records;
- preprocessing and feature config;
- package/model revisions and licenses;
- fold metrics, aggregate metrics, threshold, confusion matrices;
- context-regime metrics;
- runtime, peak RAM/VRAM, artifact size; and
- notes on failures or warnings.

Promote a candidate only when gains appear across folds/seeds and both context regimes, not from one lucky split.

## Version 1 notebook and discovery contract

- Report the observed official-sample row count dynamically; never require 299 in reusable code.
- Validate the labeled filename, schema, nonempty content, integer `{0,1}` labels, both-class presence, and the known local hash when available.
- Accept `sample submission.csv` first and `sample_submission.csv` second. Hash same-named duplicates; accept only byte-identical copies and choose deterministically.
- Treat `notebooks/generated/tfidf_baseline_v1.py` as canonical and generate the clean `.ipynb` with Jupytext.
- Scan executable notebook content for forbidden behavior without rejecting explanatory Markdown.
