# Baseline Version 1 Results

## Scope and safety

This run used only the official `data/competition/dataset samples.json` file at SHA-256 `f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28`. The observed row count was 299; reusable code does not require that count. Labels were 136 hallucinated (`0`) and 163 faithful (`1`). Normalized context was present for 130 rows and absent for 169.

No record from `data/public-20k/` was used for training, validation, vocabulary fitting, or threshold selection. No local competition test file was opened. No raw labeled example, row-level probability, or row-level prediction is stored here. The run used no external API, network call, model download, or pretrained model.

## Locked configuration

| Component | Configuration |
|---|---|
| Marked input | `__PROMPT__`, `__CONTEXT_PRESENT__`, `__CONTEXT__`, `__RESPONSE__` in fixed order |
| Preprocessing | Unicode NFC; null-like context to empty; repeated whitespace collapsed; punctuation/numbers preserved; no translation, stemming, stop-word removal, or online normalizer |
| Word TF-IDF | `analyzer="word"`, 1–2 grams, `max_features=20_000`, `min_df=1`, `max_df=1.0`, sublinear TF, case preserved, no accent stripping, `float32`, token pattern `(?u)\b\w+\b`, L2 norm |
| Character TF-IDF | `analyzer="char_wb"`, 3–5 grams, `max_features=30_000`, `min_df=1`, `max_df=1.0`, sublinear TF, case preserved, no accent stripping, `float32`, L2 norm |
| Feature combination | `FeatureUnion` |
| Classifier | `LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, class_weight="balanced", random_state=42)` |
| Fixed CV | 5-fold stratified, shuffled, seed 42 |
| Nested CV | Outer 5-fold seed 42; inner 3-fold seed `42 + outer_zero_index` |
| Threshold grid | 0.20–0.80 inclusive by 0.01 |

`min_df=1` is deliberate because the official labeled sample is small and rare Bengali terms may be informative.

## Runtime and package versions

- Local Python: 3.14.4
- NumPy: 2.5.1
- pandas: 3.0.3
- scikit-learn: 1.9.0
- End-to-end local validation plus final in-memory fit: 12.07 seconds
- Hardware path: CPU

## Trivial-predictor diagnostics

| Predictor | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix `[true rows][predicted cols]` |
|---|---:|---:|---:|---:|---|
| Always label 0 | 0.625287 | 0.000000 | 0.312644 | 0.454849 | `[[136, 0], [163, 0]]` |
| Always label 1 | 0.000000 | 0.705628 | 0.352814 | 0.545151 | `[[0, 136], [0, 163]]` |
| Majority class (label 1) | 0.000000 | 0.705628 | 0.352814 | 0.545151 | `[[0, 136], [0, 163]]` |

## 1. Fixed threshold baseline

These are standard 5-fold OOF predictions at threshold 0.50.

### Fold metrics

| Fold | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---:|---|
| 1 | 0.509091 | 0.584615 | 0.546853 | 0.550000 | `[[14, 13], [14, 19]]` |
| 2 | 0.367347 | 0.563380 | 0.465364 | 0.483333 | `[[9, 18], [13, 20]]` |
| 3 | 0.590164 | 0.576271 | 0.583218 | 0.583333 | `[[18, 9], [16, 17]]` |
| 4 | 0.423077 | 0.558824 | 0.490950 | 0.500000 | `[[11, 17], [13, 19]]` |
| 5 | 0.363636 | 0.621622 | 0.492629 | 0.525424 | `[[8, 19], [9, 23]]` |

### Fold mean and population standard deviation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Class-0 F1 | 0.450663 | 0.087350 |
| Class-1 F1 | 0.580942 | 0.022306 |
| Macro F1 | 0.515803 | 0.042913 |
| Accuracy | 0.528418 | 0.035583 |

### Aggregate fixed-threshold OOF metrics

| Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---|
| 0.459770 | 0.581602 | 0.520686 | 0.528428 | `[[60, 76], [65, 98]]` |

## 2. Honest nested macro-F1 threshold estimate

Each outer fold used a threshold selected exclusively from 3-fold inner OOF probabilities over that outer fold’s training portion.

Thresholds maximize ordinary two-class macro F1, then class-0 F1, then proximity to 0.50, then the lower threshold.

### Outer-fold results

| Outer fold | Inner-selected threshold | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.52 | 0.533333 | 0.533333 | 0.533333 | 0.533333 | `[[16, 11], [17, 16]]` |
| 2 | 0.50 | 0.367347 | 0.563380 | 0.465364 | 0.483333 | `[[9, 18], [13, 20]]` |
| 3 | 0.51 | 0.584615 | 0.509091 | 0.546853 | 0.550000 | `[[19, 8], [19, 14]]` |
| 4 | 0.52 | 0.452830 | 0.567164 | 0.509997 | 0.516667 | `[[12, 16], [13, 19]]` |
| 5 | 0.51 | 0.391304 | 0.611111 | 0.501208 | 0.525424 | `[[9, 18], [10, 22]]` |

### Outer-fold mean and population standard deviation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Class-0 F1 | 0.465886 | 0.082570 |
| Class-1 F1 | 0.556816 | 0.034430 |
| Macro F1 | 0.511351 | 0.028154 |
| Accuracy | 0.521751 | 0.022124 |

### Nested macro-F1 estimate

| Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---|
| 0.476190 | 0.560000 | 0.518095 | 0.521739 | `[[65, 71], [72, 91]]` |

This is the primary honest threshold-selected estimate while the organizer metric is ambiguous. It keeps both predicted classes active and is close to the fixed-0.50 reference.

## 3. Honest nested class-0-F1 estimate (experimental)

This preserves the original diagnostic: inner thresholds maximize class-0 F1, then macro F1, then proximity to 0.50, then the lower threshold.

### Outer-fold results

| Outer fold | Inner-selected threshold | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.69 | 0.627907 | 0.058824 | 0.343365 | 0.466667 | `[[27, 0], [32, 1]]` |
| 2 | 0.73 | 0.620690 | 0.000000 | 0.310345 | 0.450000 | `[[27, 0], [33, 0]]` |
| 3 | 0.70 | 0.620690 | 0.000000 | 0.310345 | 0.450000 | `[[27, 0], [33, 0]]` |
| 4 | 0.70 | 0.635294 | 0.114286 | 0.374790 | 0.483333 | `[[27, 1], [30, 2]]` |
| 5 | 0.74 | 0.635294 | 0.060606 | 0.347950 | 0.474576 | `[[27, 0], [31, 1]]` |

### Outer-fold mean and population standard deviation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Class-0 F1 | 0.627975 | 0.006531 |
| Class-1 F1 | 0.046743 | 0.043058 |
| Macro F1 | 0.337359 | 0.024531 |
| Accuracy | 0.464915 | 0.013271 |

### Nested class-0-F1 estimate

| Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---|
| 0.627907 | 0.047619 | 0.337763 | 0.464883 | `[[135, 1], [159, 4]]` |

The nested estimate beats the trivial always-label-0 class-0 F1 by only 0.002620 and predicts class 1 for just five OOF rows. It is therefore evidence that the provisional class-0-only objective produces a nearly degenerate classifier, not evidence of a broadly useful model.

## 4. Full-OOF threshold tuning estimates

All scores in this section are **tuning estimates** and are optimistic because the same pooled full-OOF labels selected and evaluated each threshold.

### Macro-F1-selected provisional default

Strategy `macro_f1_oof` selected threshold **0.53** by maximizing macro F1, then class-0 F1, then proximity to 0.50, then the lower threshold.

| Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---|
| 0.513699 | 0.535948 | 0.524823 | 0.525084 | `[[75, 61], [81, 82]]` |

This is the provisional deployment default because it protects performance for both labels while the exact evaluator remains ambiguous. It does not replace organizer clarification, and it must not be tuned from public leaderboard results.

### Class-0-F1-selected experimental threshold

Strategy `class0_f1_oof_experimental` selected threshold **0.71** by maximizing class-0 F1 first.

| Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy | Confusion matrix |
|---:|---:|---:|---:|---|
| 0.629371 | 0.059172 | 0.344271 | 0.468227 | `[[135, 1], [158, 5]]` |

Threshold 0.71 predicts label 0 for 98.0% of full-OOF rows. It is experimental and is not the default. Do not submit its output unless the organizers explicitly confirm binary F1 with `pos_label=0`.

### Prediction-collapse warnings

- Nested class-0-F1 selection predicts label 0 for 98.3% of rows.
- Full-OOF class-0-F1 tuning predicts label 0 for 98.0% of rows.
- Both trigger the greater-than-90% warning: a high class-specific F1 may be caused by class collapse rather than useful discrimination.
- Fixed 0.50, nested macro-F1, and full-OOF macro-F1 predictions do not trigger the warning.

### Complete full-OOF threshold table

| Threshold | Class-0 F1 | Class-1 F1 | Macro F1 | Accuracy |
|---:|---:|---:|---:|---:|
| 0.20 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.21 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.22 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.23 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.24 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.25 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.26 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.27 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.28 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.29 | 0.000000 | 0.705628 | 0.352814 | 0.545151 |
| 0.30 | 0.000000 | 0.700000 | 0.350000 | 0.538462 |
| 0.31 | 0.000000 | 0.700000 | 0.350000 | 0.538462 |
| 0.32 | 0.000000 | 0.700000 | 0.350000 | 0.538462 |
| 0.33 | 0.000000 | 0.694323 | 0.347162 | 0.531773 |
| 0.34 | 0.068966 | 0.701987 | 0.385476 | 0.548495 |
| 0.35 | 0.081633 | 0.700665 | 0.391149 | 0.548495 |
| 0.36 | 0.107383 | 0.703786 | 0.405584 | 0.555184 |
| 0.37 | 0.105960 | 0.697987 | 0.401973 | 0.548495 |
| 0.38 | 0.104575 | 0.692135 | 0.398355 | 0.541806 |
| 0.39 | 0.139241 | 0.690909 | 0.415075 | 0.545151 |
| 0.40 | 0.195122 | 0.695853 | 0.445487 | 0.558528 |
| 0.41 | 0.223529 | 0.691589 | 0.457559 | 0.558528 |
| 0.42 | 0.266667 | 0.684211 | 0.475439 | 0.558528 |
| 0.43 | 0.278075 | 0.671533 | 0.474804 | 0.548495 |
| 0.44 | 0.309278 | 0.668317 | 0.488798 | 0.551839 |
| 0.45 | 0.325123 | 0.653165 | 0.489144 | 0.541806 |
| 0.46 | 0.349057 | 0.642487 | 0.495772 | 0.538462 |
| 0.47 | 0.391304 | 0.619565 | 0.505435 | 0.531773 |
| 0.48 | 0.414938 | 0.605042 | 0.509990 | 0.528428 |
| 0.49 | 0.433735 | 0.595989 | 0.514862 | 0.528428 |
| 0.50 | 0.459770 | 0.581602 | 0.520686 | 0.528428 |
| 0.51 | 0.472325 | 0.562691 | 0.517508 | 0.521739 |
| 0.52 | 0.492857 | 0.553459 | 0.523158 | 0.525084 |
| 0.53 | 0.513699 | 0.535948 | 0.524823 | 0.525084 |
| 0.54 | 0.511475 | 0.491468 | 0.501471 | 0.501672 |
| 0.55 | 0.536741 | 0.491228 | 0.513985 | 0.515050 |
| 0.56 | 0.540373 | 0.463768 | 0.502070 | 0.505017 |
| 0.57 | 0.547059 | 0.403101 | 0.475080 | 0.484950 |
| 0.58 | 0.550143 | 0.369478 | 0.459811 | 0.474916 |
| 0.59 | 0.564384 | 0.317597 | 0.440990 | 0.468227 |
| 0.60 | 0.566845 | 0.276786 | 0.421815 | 0.458194 |
| 0.61 | 0.587629 | 0.238095 | 0.412862 | 0.464883 |
| 0.62 | 0.603015 | 0.210000 | 0.406508 | 0.471572 |
| 0.63 | 0.596577 | 0.126984 | 0.361781 | 0.448161 |
| 0.64 | 0.600484 | 0.108108 | 0.354296 | 0.448161 |
| 0.65 | 0.605769 | 0.098901 | 0.352335 | 0.451505 |
| 0.66 | 0.615752 | 0.100559 | 0.358155 | 0.461538 |
| 0.67 | 0.619385 | 0.080000 | 0.349693 | 0.461538 |
| 0.68 | 0.621176 | 0.069364 | 0.345270 | 0.461538 |
| 0.69 | 0.626168 | 0.058824 | 0.342496 | 0.464883 |
| 0.70 | 0.626168 | 0.058824 | 0.342496 | 0.464883 |
| 0.71 | 0.629371 | 0.059172 | 0.344271 | 0.468227 |
| 0.72 | 0.626450 | 0.035928 | 0.331189 | 0.461538 |
| 0.73 | 0.625000 | 0.024096 | 0.324548 | 0.458194 |
| 0.74 | 0.623557 | 0.012121 | 0.317839 | 0.454849 |
| 0.75 | 0.626728 | 0.012195 | 0.319462 | 0.458194 |
| 0.76 | 0.625287 | 0.000000 | 0.312644 | 0.454849 |
| 0.77 | 0.625287 | 0.000000 | 0.312644 | 0.454849 |
| 0.78 | 0.625287 | 0.000000 | 0.312644 | 0.454849 |
| 0.79 | 0.625287 | 0.000000 | 0.312644 | 0.454849 |
| 0.80 | 0.625287 | 0.000000 | 0.312644 | 0.454849 |

## Context-regime metrics

| Evaluation | Regime | Count | Class-0 F1 | Macro F1 | Confusion matrix |
|---|---|---:|---:|---:|---|
| Fixed 0.50 | Context absent | 169 | 0.570000 | 0.473406 | `[[57, 32], [54, 26]]` |
| Fixed 0.50 | Context present | 130 | 0.098361 | 0.410989 | `[[3, 44], [11, 72]]` |
| Nested macro-F1 | Context absent | 169 | 0.593301 | 0.467193 | `[[62, 27], [58, 22]]` |
| Nested macro-F1 | Context present | 130 | 0.093750 | 0.398916 | `[[3, 44], [14, 69]]` |
| Nested class-0 experimental | Context absent | 169 | 0.689922 | 0.344961 | `[[89, 0], [80, 0]]` |
| Nested class-0 experimental | Context present | 130 | 0.534884 | 0.312896 | `[[46, 1], [79, 4]]` |
| Full-OOF macro threshold 0.53 | Context absent | 169 | 0.637168 | 0.452513 | `[[72, 17], [65, 15]]` |
| Full-OOF macro threshold 0.53 | Context present | 130 | 0.090909 | 0.390815 | `[[3, 44], [16, 67]]` |
| Full-OOF class-0 threshold 0.71 | Context absent | 169 | 0.689922 | 0.344961 | `[[89, 0], [80, 0]]` |
| Full-OOF class-0 threshold 0.71 | Context present | 130 | 0.538012 | 0.325186 | `[[46, 1], [78, 5]]` |

Both context subgroups contain both classes. Every strategy remains especially weak for context-present class-0 examples. The class-0-selected high thresholds improve class-0 F1 largely by predicting almost everything as hallucinated.

## Known limitations and interpretation

1. The official evaluator wording remains ambiguous between class-0 binary F1 and two-class macro F1; organizer clarification remains necessary.
2. An all-zero predictor already receives class-0 F1 0.625287 on this label distribution. Class-0-only optimization therefore selects a nearly all-zero model and collapses class-1 F1.
3. Macro-F1 thresholding is the provisional default because it rewards useful discrimination across both classes. Threshold 0.71 remains experimental.
4. Public leaderboard threshold probing is prohibited. The leaderboard cannot replace official metric clarification.
5. The official labeled sample is very small; fold variance is material and there is no separate untouched local holdout.
6. The fixed model has little evidence of useful context grounding, particularly for context-present hallucinations.
7. Both full-OOF selected scores are optimistic tuning estimates; the corresponding nested results are the honest threshold-tuned estimates.
8. This is a pipeline and submission-safety baseline, not a competitive final model.

## Reasonable Kaggle submissions

The two reasonable Version 1 candidates are the macro-F1-selected default `submission.csv` and the clean fixed-threshold reference `submission_fixed_050.csv`. Do not recommend or submit `submission_class0_experimental.csv` unless the official evaluator is explicitly confirmed as binary F1 with `pos_label=0`.

## Reproducibility confirmations

- Only official labeled records were used.
- The public dataset and its copied test file were not opened by the training/validation path.
- The real local competition test CSV was not opened.
- No submission was generated locally.
- No external API, web request, model download, transformer, or pretrained weight was used.
- The original starter notebook remained unchanged.
