# V4-A artifact and reproducibility audit

Status: authenticated aggregate-only audit complete. No V6 training, test scoring,
submission generation, threshold search, commit, or push was performed.

## Artifact authentication

The teammate files were moved without byte changes into the ignored local directory
`artifacts/shared/v4a_0685/`.

| File | Bytes | SHA-256 | Modified (UTC) |
|---|---:|---|---|
| `v4a_oof_probabilities.csv` | 8,076 | `78ee80a3f9944500090342170ccf60db7cbc793c7fbe6a17ad3e6a6906d51ea6` | 2026-07-18 12:08:08.3021502 |
| `v4a_test_probabilities.csv` | 60,029 | `2dd5ffa97c285727143fefe7d172bd02588368b7c3ff111088709ce0eb27adc2` | 2026-07-18 12:08:20.6645779 |
| `v4a_run_summary.json` | 8,783 | `b30dabf16cacf0bb684791740cf1d486b73bade27ff6805edb9daf0bccffca85` | 2026-07-18 12:08:13.9749058 |

All three files are untracked and ignored. The summary does not contain teammate-provided
artifact hashes, so the hashes above are the local authentication baseline.

## Schemas

### OOF probabilities

- columns: `row_index`, `fold`, `label`, `probability_label1`;
- rows: 299;
- types: all integer except `probability_label1` (`float64`);
- missing values: zero in every column;
- unique source indices: 299; duplicate source indices: zero;
- probability minimum/maximum/mean/population standard deviation:
  0.0324415009 / 0.9606169724 / 0.5137349737 / 0.2627355651;
- fold values: 0, 1, 2, 3, 4, with 59, 60, 61, 60, and 59 rows;
- validation-fold label counts `[label 0, label 1]`: `[24,35]`, `[19,41]`,
  `[29,32]`, `[34,26]`, and `[30,29]`.

The probability column explicitly identifies label 1. Its semantic orientation is
compatible with the authenticated repository convention, label 1 = faithful, and the
artifact labels exactly equal the official labels. The summary does not provide a separate
natural-language orientation declaration.

Exactly one finite OOF probability exists for every official source index from 0 through
298. There are no missing, duplicate, or out-of-range aligned rows.

### Test probabilities

- columns: `id`, `probability_label1`;
- rows: 2,516, matching the summary and expected count;
- types: `id` is integer and probability is `float64`;
- missing values: zero;
- unique IDs: 2,516; duplicate IDs: zero;
- probability minimum/maximum/mean/population standard deviation:
  0.0000657057 / 0.9832827949 / 0.4984692497 / 0.2671775352.

The summary declares the two columns and row count, but contains no ID-order hash or other
alignment checksum. Test IDs and individual probabilities were not printed, scored,
combined, or used for selection.

### Run summary

Top-level keys are `artifacts`, `configuration`, `environment`, `group_audit`,
`honest_grouped_estimate`, `official_labeled_count`, `original_method_estimate`,
`runtime_seconds`, `source_method_verified_kaggle_version`,
`source_method_verified_public_score`, `status`, `test_count`,
`test_prediction_collapse_warning`, `test_prediction_distribution`, `version`, and
`wikipedia`.

Recorded configuration:

- Wikipedia source: `abyaadrafid/bnwiki`;
- usable articles: 62,150;
- parsed canonical chunks: 301;
- deployed retrieval threshold: 0.25;
- classifier threshold: 0.50;
- random state: 42;
- evidence snippet limit: 800 characters;
- retrieval vocabulary maximum: 50,000 features;
- classifier features: `ctx_present`, `word_overlap_ratio`, `numbers_supported`,
  `number_overlap_ratio`, `has_numbers_in_response`, `is_substring`,
  `response_len_chars`, and `context_len_chars`;
- validation: nested 5x3 duplicate-aware grouped estimate across all official rows;
- outer selected retrieval thresholds: 0.20, 0.40, 0.20, 0.20, 0.30;
- group audit: 296 groups, three nontrivial groups, largest group two, zero exact-duplicate
  links, and three prompt/context links;
- recorded test prediction counts: label 0 = 1,664 and label 1 = 852;
- recorded runtime: 5,113.533819 seconds.

## OOF metric reproduction at threshold 0.50

The OOF indices were sorted and joined to the authenticated official labels by
`row_index`. Artifact labels exactly matched the official labels. Reproduced metrics were:

| Metric | Value |
|---|---:|
| Macro F1 | 0.6872701508 |
| Label-0 F1 | 0.7102803738 |
| Label-1 F1 | 0.6642599278 |
| Accuracy | 0.6889632107 |
| Confusion matrix | `[[114,22],[71,92]]` |
| Predicted counts `[0,1]` | `[185,114]` |

The maximum absolute difference from the declared scalar metrics was
`3.18e-11`, passing the strict `1e-9` tolerance.

## Corrected route audit

Corrected routing produced 130 context-present and 169 context-absent rows.

| Metric | Context present | Context absent |
|---|---:|---:|
| Macro F1 | 0.9246667954 | 0.4339712919 |
| Label-0 F1 | 0.9032258065 | 0.6315789474 |
| Label-1 F1 | 0.9461077844 | 0.2363636364 |
| Accuracy | 0.9307692308 | 0.5029585799 |
| Confusion matrix | `[[42,5],[4,79]]` | `[[72,17],[67,13]]` |
| Predicted counts `[0,1]` | `[46,84]` | `[139,30]` |
| Brier score | 0.0811956466 | 0.2801883321 |
| Probability mean | 0.6357301996 | 0.4198924922 |
| Probability population std. | 0.3116398002 | 0.1648272116 |

V4-A's value comes primarily from context-present performance, not null-route retrieval.
Its context-present macro F1 is 0.024641 above the frozen substring rule's 0.900026, while
its context-absent macro F1 is 0.107792 below the handcrafted lexical champion's 0.541763.
The null route strongly underpredicts label 1 and is poorly calibrated relative to the
present route.

## Fixed route-aware hybrid diagnostic

Diagnostic name: `v5_substring_v4a_retrieval_hybrid_fixed050`.

The deterministic rule was fixed before evaluation:

- corrected context-present rows: frozen Candidate A hard prediction;
- corrected context-absent rows: V4-A `probability_label1 >= 0.50`.

No threshold search was performed.

| Metric | Value |
|---|---:|
| Macro F1 | 0.6770186335 |
| Label-0 F1 | 0.7018633540 |
| Label-1 F1 | 0.6521739130 |
| Accuracy | 0.6789297659 |
| Confusion matrix | `[[113,23],[73,90]]` |
| Predicted counts `[0,1]` | `[186,113]` |
| Context-present macro F1 | 0.9000256345 |
| Context-absent macro F1 | 0.4339712919 |

This is 0.015032 below the current routed champion (0.692051) and 0.010252 below
V4-A's full OOF result (0.6872701508). Replacing V4-A's strong present-route classifier
with the substring rule exposes the weakness of its null-route retrieval predictions.

## Fold and future-V6 compatibility

V4-A supplies a single five-fold assignment. The summary describes honest nested 5x3
grouped validation, but the artifacts contain no outer-training memberships, inner-fold
memberships, fitted-corpus fingerprints, or group IDs. Therefore exclusion of validation
rows from each fitted model is documented but cannot be independently reconstructed from
the preserved artifacts alone.

The strengthened V5/V6 audit has 291 groups. V4-A splits three of those strengthened
groups across different folds. Its partition does not match the V5/V6 partition for seed
17, 29, or 43, even when fold numbers are treated as arbitrary labels. Consequently:

- the existing V4-A and future V6 OOF probabilities will align by official row and label
  orientation;
- their fold-generating processes are not compatible;
- a free blend or stack evaluated on these existing OOF values would not have a shared
  unbiased outer-fold interpretation;
- route-level replacement is safer than choosing blend weights from the mismatched OOF
  artifacts;
- an unbiased thresholded V4-A hybrid would require regenerating retrieval OOF values
  within each strengthened outer-training partition, with threshold selection restricted
  to corresponding inner/training OOF data.

No nested threshold selection was executed because the preserved V4-A folds are
incompatible with the strengthened validation design.

## Recommendation and unresolved risks

Recommendation: **2. Run V6 and compare without blending.**

The V4-A null route is substantially below the current handcrafted null champion, and its
folds are incompatible with V5/V6. V6 should therefore be evaluated under its already
prepared strengthened repeated grouped protocol and compared route-wise. Do not blend V4-A
and V6 probabilities unless V4-A is later regenerated on compatible outer folds.

Unresolved risks:

- V4-A training membership cannot be authenticated from the aggregate artifacts;
- V4-A splits three strengthened groups;
- test ID order has no independent alignment hash;
- the summary has no artifact checksums;
- retrieval corpus and chunk identity are summarized but not cryptographically recorded;
- the public score is contextual evidence only and was not used for tuning in this audit.
