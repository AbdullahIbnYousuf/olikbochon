# Version 5 Official Lexical Baseline Plan

## Scope and status

Experiment identity: `v5_official_lexical_baseline`.

This is an official-labeled-sample-only, non-neural experiment. The authenticated sample is
loaded through the existing hash-checking loader. Competition-test data, public 5K/20K data,
leaderboard feedback, neural inference, lookup tables, and manually labeled examples are not
inputs. Preparation is complete, but the grouped candidate evaluation is not authorized yet.

The corrected schema route is retained: 130 context-present rows and 169 context-absent rows.
The audit has 47 label-0 and 83 label-1 context-present rows, plus 89 label-0 and 80 label-1
context-absent rows. Label 0 means hallucinated; label 1 means faithful.

## Frozen normalization

`normalize_v5` is shared by audit, training, and future inference. It applies Unicode NFC,
removes zero-width formatting characters, maps Bengali/Arabic-Indic/Eastern-Arabic-Indic
digits to ASCII digit equivalents, maps fixed punctuation/quotation variants to canonical
ASCII forms, lowercases Latin characters only, collapses repeated whitespace, and trims.
Bengali words, named entities, and numerals are not removed. Punctuation-stripped matching
replaces Unicode punctuation with spaces after this normalization.

Tokens are deterministic contiguous Unicode letter/mark/numeral sequences. Character
similarity uses punctuation-stripped, whitespace-free normalized strings. Longest common
subsequence and substring ratios use the response character length as denominator. Token and
character coverage use multiset intersection divided by response size. Empty-to-empty
Jaccard/coverage is 1.0; empty response alignment is never treated as an exact substring.

Rare tokens have response-corpus frequency at most 2. The read-only audit fits this frequency
map on the official sample solely for aggregate exploration. Every grouped model fold fits its
frequency map again using only that route's training subset, so validation frequencies cannot
leak into model features.

## Frozen feature lists

Context-present logistic regression uses these 27 features, in this order:

1. `response_in_context_exact`
2. `response_in_context_no_punctuation`
3. `context_in_response_exact`
4. `response_token_coverage`
5. `response_character_coverage`
6. `token_jaccard`
7. `char3_jaccard`
8. `char4_jaccard`
9. `longest_common_subsequence_ratio`
10. `longest_common_substring_ratio`
11. `response_context_length_ratio`
12. `response_char_count`
13. `response_token_count`
14. `prompt_char_count`
15. `prompt_token_count`
16. `context_char_count`
17. `context_token_count`
18. `numeric_consistency_ratio`
19. `numeral_mismatch_count`
20. `response_numeral_count`
21. `response_bengali_numeral_count`
22. `response_arabic_numeral_count`
23. `context_bengali_numeral_count`
24. `context_arabic_numeral_count`
25. `rare_token_coverage`
26. `prompt_response_token_jaccard`
27. `prompt_context_token_jaccard`

Context-absent logistic regression uses these 21 features, in this order:

1. `response_char_count`
2. `response_token_count`
3. `prompt_char_count`
4. `prompt_token_count`
5. `prompt_response_token_jaccard`
6. `prompt_response_char3_jaccard`
7. `response_numeral_count`
8. `response_bengali_numeral_count`
9. `response_arabic_numeral_count`
10. `response_year_like_count`
11. `response_punctuation_count`
12. `response_negation_count`
13. `response_uncertainty_count`
14. `response_bengali_script_ratio`
15. `response_latin_script_ratio`
16. `prompt_has_question_mark`
17. `prompt_interrogative_count`
18. `prompt_instruction_count`
19. `response_rare_token_count`
20. `response_named_entity_like_count`
21. `response_specificity_score`

The specificity score is the sum of response numeral, rare-token, and orthographic
named-entity-like token counts. Entity-like is deliberately low-capacity: Latin title/acronym
tokens or mixed letter-digit tokens. It is not an external NER system.

## Read-only official feature audit

All entries below are `count / mean / median / population SD / minimum / maximum`. These are
aggregate exploratory statistics only; no raw text or row-level values were printed or saved.

### Context-present rows

| Feature | Label 0 | Label 1 |
|---|---:|---:|
| response_in_context_exact | 47 / 0.127660 / 0.000000 / 0.333710 / 0.000000 / 1.000000 | 83 / 0.927711 / 1.000000 / 0.258966 / 0.000000 / 1.000000 |
| response_in_context_no_punctuation | 47 / 0.148936 / 0.000000 / 0.356026 / 0.000000 / 1.000000 | 83 / 0.927711 / 1.000000 / 0.258966 / 0.000000 / 1.000000 |
| context_in_response_exact | 47 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 | 83 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |
| response_token_coverage | 47 / 0.589805 / 0.600000 / 0.310171 / 0.000000 / 1.000000 | 83 / 0.925703 / 1.000000 / 0.251173 / 0.000000 / 1.000000 |
| response_character_coverage | 47 / 0.969145 / 1.000000 / 0.046953 / 0.750000 / 1.000000 | 83 / 0.986101 / 1.000000 / 0.109543 / 0.000000 / 1.000000 |
| token_jaccard | 47 / 0.051862 / 0.041667 / 0.051639 / 0.000000 / 0.333333 | 83 / 0.040431 / 0.028169 / 0.036315 / 0.000000 / 0.173913 |
| char3_jaccard | 47 / 0.057064 / 0.048632 / 0.056104 / 0.002475 / 0.358491 | 83 / 0.033937 / 0.022654 / 0.037499 / 0.000000 / 0.175926 |
| char4_jaccard | 47 / 0.043424 / 0.032500 / 0.051177 / 0.000000 / 0.336364 | 83 / 0.028243 / 0.017606 / 0.034464 / 0.000000 / 0.159292 |
| longest_common_subsequence_ratio | 47 / 0.897543 / 0.923077 / 0.117006 / 0.534884 / 1.000000 | 83 / 0.982958 / 1.000000 / 0.115191 / 0.000000 / 1.000000 |
| longest_common_substring_ratio | 47 / 0.512386 / 0.441176 / 0.279565 / 0.131579 / 1.000000 | 83 / 0.948998 / 1.000000 / 0.190582 / 0.000000 / 1.000000 |
| response_context_length_ratio | 47 / 0.096243 / 0.052071 / 0.117880 / 0.012695 / 0.545455 | 83 / 0.031347 / 0.019139 / 0.031742 / 0.001203 / 0.130435 |
| response_char_count | 47 / 35.170213 / 32.000000 / 19.234631 / 7.000000 / 88.000000 | 83 / 13.240964 / 11.000000 / 9.265287 / 1.000000 / 47.000000 |
| response_token_count | 47 / 5.553191 / 5.000000 / 2.908597 / 1.000000 / 13.000000 | 83 / 2.325301 / 2.000000 / 1.489792 / 1.000000 / 7.000000 |
| prompt_char_count | 47 / 50.489362 / 44.000000 / 19.074180 / 23.000000 / 120.000000 | 83 / 50.831325 / 46.000000 / 18.011254 / 22.000000 / 113.000000 |
| prompt_token_count | 47 / 7.531915 / 6.000000 / 2.774129 / 4.000000 / 18.000000 | 83 / 7.289157 / 7.000000 / 2.519945 / 3.000000 / 17.000000 |
| context_char_count | 47 / 616.127660 / 556.000000 / 398.044508 / 115.000000 / 1733.000000 | 83 / 603.168675 / 553.000000 / 368.720739 / 66.000000 / 2173.000000 |
| context_token_count | 47 / 90.191489 / 78.000000 / 58.061464 / 17.000000 / 255.000000 | 83 / 89.746988 / 81.000000 / 55.920736 / 12.000000 / 347.000000 |
| numeric_consistency_ratio | 47 / 0.797872 / 1.000000 / 0.394909 / 0.000000 / 1.000000 | 83 / 0.975904 / 1.000000 / 0.153348 / 0.000000 / 1.000000 |
| numeral_mismatch_count | 47 / 0.255319 / 0.000000 / 0.524631 / 0.000000 / 2.000000 | 83 / 0.024096 / 0.000000 / 0.153348 / 0.000000 / 1.000000 |
| response_numeral_count | 47 / 1.574468 / 0.000000 / 2.039448 / 0.000000 / 6.000000 | 83 / 1.578313 / 0.000000 / 2.168340 / 0.000000 / 8.000000 |
| response_bengali_numeral_count | 47 / 1.574468 / 0.000000 / 2.039448 / 0.000000 / 6.000000 | 83 / 1.578313 / 0.000000 / 2.168340 / 0.000000 / 8.000000 |
| response_arabic_numeral_count | 47 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 | 83 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |
| context_bengali_numeral_count | 47 / 11.553191 / 9.000000 / 10.286965 / 0.000000 / 50.000000 | 83 / 11.927711 / 10.000000 / 9.525847 / 0.000000 / 40.000000 |
| context_arabic_numeral_count | 47 / 0.255319 / 0.000000 / 1.731659 / 0.000000 / 12.000000 | 83 / 0.698795 / 0.000000 / 3.060573 / 0.000000 / 22.000000 |
| rare_token_coverage | 47 / 0.833486 / 0.857143 / 0.162803 / 0.500000 / 1.000000 | 83 / 0.846816 / 1.000000 / 0.258650 / 0.000000 / 1.000000 |
| prompt_response_token_jaccard | 47 / 0.253142 / 0.125000 / 0.267229 / 0.000000 / 0.800000 | 83 / 0.011847 / 0.000000 / 0.039684 / 0.000000 / 0.166667 |
| prompt_context_token_jaccard | 47 / 0.058332 / 0.041096 / 0.055499 / 0.000000 / 0.333333 | 83 / 0.060805 / 0.050000 / 0.054602 / 0.000000 / 0.411765 |

### Context-absent rows

| Feature | Label 0 | Label 1 |
|---|---:|---:|
| response_char_count | 89 / 15.056180 / 13.000000 / 12.689339 / 1.000000 / 67.000000 | 80 / 13.362500 / 11.000000 / 10.768291 / 1.000000 / 79.000000 |
| response_token_count | 89 / 2.528090 / 2.000000 / 2.000505 / 0.000000 / 10.000000 | 80 / 2.325000 / 2.000000 / 1.794262 / 0.000000 / 13.000000 |
| prompt_char_count | 89 / 53.000000 / 42.000000 / 30.613209 / 11.000000 / 156.000000 | 80 / 49.837500 / 46.500000 / 23.884851 / 18.000000 / 137.000000 |
| prompt_token_count | 89 / 8.820225 / 7.000000 / 5.247925 / 2.000000 / 29.000000 | 80 / 8.125000 / 7.000000 / 4.160454 / 3.000000 / 21.000000 |
| prompt_response_token_jaccard | 89 / 0.033961 / 0.000000 / 0.062317 / 0.000000 / 0.333333 | 80 / 0.027513 / 0.000000 / 0.058719 / 0.000000 / 0.250000 |
| prompt_response_char3_jaccard | 89 / 0.031071 / 0.000000 / 0.066901 / 0.000000 / 0.387097 | 80 / 0.036496 / 0.000000 / 0.067543 / 0.000000 / 0.244444 |
| response_numeral_count | 89 / 0.662921 / 0.000000 / 1.506121 / 0.000000 / 8.000000 | 80 / 0.862500 / 0.000000 / 1.882444 / 0.000000 / 8.000000 |
| response_bengali_numeral_count | 89 / 0.629213 / 0.000000 / 1.501924 / 0.000000 / 8.000000 | 80 / 0.775000 / 0.000000 / 1.830130 / 0.000000 / 8.000000 |
| response_arabic_numeral_count | 89 / 0.033708 / 0.000000 / 0.234614 / 0.000000 / 2.000000 | 80 / 0.087500 / 0.000000 / 0.574320 / 0.000000 / 5.000000 |
| response_year_like_count | 89 / 0.044944 / 0.000000 / 0.207181 / 0.000000 / 1.000000 | 80 / 0.075000 / 0.000000 / 0.263391 / 0.000000 / 1.000000 |
| response_punctuation_count | 89 / 0.415730 / 0.000000 / 0.731460 / 0.000000 / 4.000000 | 80 / 0.237500 / 0.000000 / 0.575407 / 0.000000 / 3.000000 |
| response_negation_count | 89 / 0.044944 / 0.000000 / 0.207181 / 0.000000 / 1.000000 | 80 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |
| response_uncertainty_count | 89 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 | 80 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |
| response_bengali_script_ratio | 89 / 0.769101 / 1.000000 / 0.418095 / 0.000000 / 1.000000 | 80 / 0.887500 / 1.000000 / 0.315981 / 0.000000 / 1.000000 |
| response_latin_script_ratio | 89 / 0.096067 / 0.000000 / 0.289927 / 0.000000 / 1.000000 | 80 / 0.037500 / 0.000000 / 0.189984 / 0.000000 / 1.000000 |
| prompt_has_question_mark | 89 / 0.876404 / 1.000000 / 0.329120 / 0.000000 / 1.000000 | 80 / 0.900000 / 1.000000 / 0.300000 / 0.000000 / 1.000000 |
| prompt_interrogative_count | 89 / 0.370787 / 0.000000 / 0.505743 / 0.000000 / 2.000000 | 80 / 0.325000 / 0.000000 / 0.468375 / 0.000000 / 1.000000 |
| prompt_instruction_count | 89 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 | 80 / 0.000000 / 0.000000 / 0.000000 / 0.000000 / 0.000000 |
| response_rare_token_count | 89 / 2.202247 / 2.000000 / 1.885365 / 0.000000 / 10.000000 | 80 / 1.937500 / 2.000000 / 1.511156 / 0.000000 / 11.000000 |
| response_named_entity_like_count | 89 / 0.123596 / 0.000000 / 0.493104 / 0.000000 / 3.000000 | 80 / 0.025000 / 0.000000 / 0.156125 / 0.000000 / 1.000000 |
| response_specificity_score | 89 / 2.988764 / 2.000000 / 2.382022 / 0.000000 / 11.000000 | 80 / 2.825000 / 2.000000 / 2.443230 / 0.000000 / 11.000000 |

The strongest descriptive context-present separation is the exact response-in-context rate:
0.127660 for label 0 versus 0.927711 for label 1. This is an audit observation, not a
grouped OOF result. Several absent-route markers are constant or nearly constant and are kept
because the feature list was frozen before evaluation; `StandardScaler` safely maps constant
training-fold columns to zero.

## Strengthened grouped validation

V5 starts from existing V3 exact, prompt/context family, and character-TF-IDF near-duplicate
groups, then takes the transitive union with V5-normalized exact-row identity, prompt identity,
and context identity for context-present rows. Absent sentinels are never treated as one shared
context group. Exact V5-normalized rows with conflicting labels stop the run.

The official-only feasibility audit found 291 groups, 8 non-singleton groups, largest group 2,
and 0 conflicting V5 exact rows. Added graph links were 4 prompt-identity and 5 present-context
links; the inherited V3 graph contained 3 prompt/context links and 1 near-duplicate pair.

Five folds are feasible for seeds 17, 29, and 43: 15 total folds, 0 train/validation group
overlap, complete per-seed OOF coverage, and both labels in both routes of every train and
validation split. Validation folds contain 59–61 rows and 58–59 groups. Context-present
validation subsets contain 19–31 rows (label-0 5–12; label-1 11–19); context-absent subsets
contain 29–40 rows (label-0 15–22; label-1 14–21). There is no random fallback.

## Frozen candidates

- Candidate A, `deterministic_substring_rule`: context-present predicts label 1 only when a
  nonempty V5-normalized response is an exact substring of V5-normalized context. Context-absent
  predicts that fold's context-absent training majority; exact ties predict label 0. Diagnostic
  only.
- Candidate B, `context_present_logistic`: the 27 context features, fitted only on
  context-present training rows.
- Candidate C, `context_absent_logistic`: the 21 absent features, fitted only on
  context-absent training rows.
- Candidate D, `routed_lexical_logistic`: B for schema-defined context-present rows and C for
  context-absent rows; there is no learned route classifier.

B–D use `StandardScaler` and `LogisticRegression(class_weight="balanced", C=1.0,
max_iter=2000, random_state=42, solver="liblinear")`. No TF-IDF, trees, SVM, gradient boosting,
neural logits, or ensembles are included.

Primary reporting uses threshold 0.50. Secondary selection uses 0.20–0.80 inclusive in 0.02
increments with the existing macro-F1, label-0-F1, proximity-to-0.50, then lower-threshold
ordering; thresholds with either predicted class above 90% are ineligible. Fold/seed/route and
aggregate records contain macro F1, both label F1 values, accuracy, confusion matrix, and
predicted-class counts. Logistic candidates also report Brier/probability diagnostics,
standardized coefficients, coefficient signs, top directions, and sign stability.

## Acceptance criteria

- Substring rule: promising at context-present macro F1 at least 0.75; strong at 0.85.
- Context-present logistic: pass at least +0.01 over substring; retain without gain only for a
  material class-balance improvement.
- Context-absent logistic: pass at macro F1 at least 0.55; otherwise weak, with no further
  overfitting.
- Routed model: strong at aggregate grouped OOF macro F1 at least 0.62; high-value at 0.68;
  reject below the corrected BanglaBERT baseline 0.565096.

These thresholds are frozen before evaluation.

## Artifact and execution contract

The runner accepts no input-data or model path. It resolves only the authenticated official
sample. The exact new output must be under ignored `artifacts/v5/official_lexical_baseline/`
and must not already exist. It writes only frozen configuration and aggregate/fold metrics,
aggregate feature statistics, and coefficient reports. Raw text, identifiers, row-level
feature tables, probabilities, predictions, model files, and submissions are not persisted.

Pending approval, the exact command is:

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"

.\.venv\Scripts\python.exe -m olikbochon.v5_runner `
  --mode lexical-baseline `
  --candidate-set frozen `
  --seeds 17 29 43 `
  --folds 5 `
  --output-dir "F:\datathon\olikbochon\artifacts\v5\official_lexical_baseline"
```

Based on the completed aggregate audit, expected CPU runtime is under two minutes and expected
aggregate-only JSON storage is under 5 MB. No GPU or checkpoint storage is required.
