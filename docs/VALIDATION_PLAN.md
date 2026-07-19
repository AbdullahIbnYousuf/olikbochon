# Validation Plan

## Objectives

Validation must estimate generalization to unseen official items, support threshold selection without touching competition test text, reveal context-regime failures, and remain cheap enough for a beginner team to repeat.

The official metric wording is ambiguous. Until clarified, every run must report both class-specific F1 values and macro F1. Ordinary two-class macro F1 is the provisional decision metric because class-0-only optimization demonstrably collapses toward a trivial all-zero predictor.

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

For logistic regression, obtain probabilities and locate the label-1 column from `model.classes_`; never assume a column index. Predict class 1 when its probability is at least threshold `t`, otherwise predict class 0.

Version 1 supports three named strategies:

- `fixed_050`: threshold 0.50 and no selection;
- `macro_f1_oof`: maximize ordinary two-class macro F1, then class-0 F1, then closeness to 0.50, then the lower threshold; this is the provisional default;
- `class0_f1_oof_experimental`: maximize class-0 F1, then macro F1, then closeness to 0.50, then the lower threshold; this is never the default while the metric is ambiguous.

The locked honest procedure runs nested threshold evaluation separately for both selected strategies:

1. create outer 5-fold stratified CV with shuffle and seed 42;
2. inside outer fold `i`, create inner 3-fold stratified CV over only the outer-training records with seed `42 + i`;
3. generate inner OOF probabilities and select `t` from 0.20 through 0.80 by 0.01;
4. rank thresholds using the active strategy’s ordering;
5. fit on the complete outer-training portion and score the untouched outer-validation portion at that independently selected threshold;
6. aggregate the five outer-validation predictions as either the nested macro-F1 estimate or nested experimental class-0-F1 estimate; and
7. report each outer threshold plus fold and aggregate metrics.

Separately, generate ordinary full 5-fold OOF probabilities. Report threshold-0.50 metrics as the fixed baseline. Select both macro-F1 and experimental class-0-F1 thresholds from the same full-OOF table. Label both scores “Full-OOF ... threshold tuning estimate.” These same-OOF scores are optimistic and are not the primary honest threshold-tuned estimates.

Use the full-OOF macro-F1 threshold for `/kaggle/working/submission.csv`; also generate the clean fixed-0.50 reference. The class-0-selected file is experimental and must not be recommended unless organizers explicitly confirm binary F1 with `pos_label=0`. Once the evaluator is clarified, change the objective explicitly and version the config.

For every threshold prediction set, warn when either predicted class exceeds 90% of all predictions. The warning must state that high class-specific F1 may be caused by class collapse rather than useful discrimination. Also report aggregate all-zero, all-one, and majority-class metrics without retaining row-level predictions. On the current sample, all-zero already receives approximately 0.6253 class-0 F1, explaining why class-0-only selection is unsafe.

For a linear SVM, tune on signed decision scores. For ensembles, calibrate/blend on training-fold predictions only and repeat the same outer evaluation.

## Final untouched holdout

- If only the small official sample is usable, the locked nested CV is more informative than permanently sacrificing a large holdout; clearly state that no untouched local holdout exists.
- If the public data is approved, keep the entire official sample untouched through model/feature/threshold selection where feasible.
- Never use the competition test or public leaderboard to choose preprocessing, features, models, ensemble weights, or thresholds. Public leaderboard threshold probing is prohibited.

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

## Version 2 public-split contract

- Fit the unchanged model on the approved public 4k split and select one macro-F1 threshold on the public 1k split.
- Call the selected public score `Public-validation threshold-tuning estimate`; it is not independent.
- Keep the official labeled sample untouched until `Independent official-sample evaluation`, then apply fixed 0.50 and the already frozen selected threshold.
- Use a text-only normalized fingerprint for overlap/conflict detection and a label-inclusive fingerprint for complete-row deduplication.
- Before modeling, audit character-TF-IDF near duplicates across public train/validation and combined public/official at cosine 0.97 plus length ratio 0.90. Stop on conflicts, more than 1% affected validation rows, or more than 1% affected official rows.
- Preserve holdout rows and remove only lower-precedence public rows when a small passing set of same-label near duplicates exists.
- After evaluation, refit on every unique allowed row. Do not retune the threshold even though adding validation/official rows may shift calibration.
- Quarantine all CSVs below directories containing the public 4k/1k files. Resolve test inference only from a separate root co-locating the official sample, test, and sample submission.

## Version 3 grouped transformer contract

- Authenticate the known public and official labeled hashes and report row
  counts dynamically. The public aggregate is excluded whenever the separate
  4k/1k roles are used.
- Build official groups from exact normalized text, normalized prompt/context
  fingerprints, and the predeclared character-TF-IDF near-duplicate rule at
  cosine 0.97 plus length ratio 0.90. Related rows must remain in one fold.
- Use five deterministic `StratifiedGroupKFold` folds when any group is
  nontrivial. Plain `StratifiedKFold` is permitted only after the audit proves
  every group is a singleton. Validate complete one-time OOF placement, both
  classes in every validation fold, and zero train/validation group overlap.
- Use the identical frozen fold assignment for Arm A and Arm B. Arm A starts
  every fold from the authenticated base; Arm B starts every fold from the same
  public-selected Stage A checkpoint. No fold inherits another fold's state.
- Compare arms only at threshold 0.50. Promote Arm B only for macro gain at
  least 0.01, no predicted class above 90%, and class-0 F1 no more than 0.02
  below Arm A. Arm A wins every tie or failed guard.
- Tune the selected arm on 0.20–0.80 by 0.01 using macro F1, class-0 F1,
  closeness to 0.50, then lower threshold. Deploy it only for at least 0.01
  macro gain, range 0.40–0.60, and no class collapse; otherwise deploy 0.50.
- Label arm comparison `Official OOF model-selection estimate` and threshold
  scoring `Official OOF threshold-tuning estimate`. Both are optimistic, not
  independent performance estimates.
- Freeze the arm, threshold, configuration, and folds before final fitting or
  competition-test inference. A CUDA OOM may only trigger one complete phase
  restart under the locked smaller-batch configuration.

## Version 4-A retrieval-reproduction contract

- Authenticate official competition hashes before loading rows. Dynamically find
  the `abyaadrafid/bnwiki` AA/AB/AC/AD layout, require at least 250 `wiki_*`
  chunks, discard only byte-identical duplicate layouts, strictly parse every
  selected chunk, and record the discovered count and aggregate bytes.
- Report the public source method's ordinary no-context five-fold OOF cutoff
  score as an explicitly optimistic `Original-method ... tuning estimate`.
- Independently group exact normalized rows and every response sharing a
  normalized prompt/context. Keep each family wholly inside one fold.
- For the honest estimate, use five outer grouped folds. Select the Wikipedia
  retrieval cutoff only through three-fold grouped CV inside each outer-training
  portion, then apply it once to the untouched outer-validation fold.
- Preserve the public method's predeclared 0.25 retrieval cutoff for final
  reproduction artifacts; do not choose it from the leaderboard.
- Save honest OOF and ID-aligned test label-1 probabilities only as ignored
  Kaggle runtime artifacts for a future, separately specified V3 ensemble.
- Test data may be loaded only after retrieval, validation, feature, classifier,
  and cutoff decisions are frozen. Never display identifiers, test text,
  retrieved passages, probabilities, or individual predictions.
