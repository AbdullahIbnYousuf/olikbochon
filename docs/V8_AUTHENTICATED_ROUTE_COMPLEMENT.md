# V8 authenticated route-complement experiment

Experiment: `v8_authenticated_route_complement`.

Status: **complete and rejected under the frozen gate**. The experiment used only the
authenticated 299-row official labeled sample and the authenticated unlabeled Bengali
Wikipedia corpus. No competition-test row, V4-A test probability, teammate V4-A OOF value,
public label, leaderboard-derived decision, API, manual row review, test prediction, or
submission was used.

## Frozen design

The strengthened grouped folds were fixed at seeds 17, 29, and 43 with five folds per seed.
Candidate Q-authenticated uses threshold 0.50 without averaging, stacking, threshold search,
or a learned route selector:

- 130 context-present rows: source-regenerated V4-A-style OOF probability;
- 169 context-absent rows: source-regenerated Sparse Candidate I OOF probability.

The V4-A-style retrieval index was fit once on corpus text only: `char_wb` TF-IDF 2–4
grams, 50,000 maximum features, top-1 prompt retrieval for absent rows, cutoff 0.25, and the
first 800 normalized article characters. Present rows used supplied official context
directly. The eight frozen evidence features fed fold-local `StandardScaler` and balanced
liblinear logistic regression with C=1.0, 3,000 maximum iterations, and random state 42.

The teammate summary omits scaler metadata. Standardization was chosen from repository
evidence rather than guessed: the predeclared V7 Candidate M is explicitly documented as
standardized V4-A parity. This discrepancy is retained as a reproduction limitation.

Candidate I used its evaluated character/word TF-IDF channels, 21 numeric features,
fold-local numeric scaler and rare-token map, and the same balanced logistic configuration.
It fitted only absent-route outer-training rows.

## Corpus and index authentication

- logical manifest: `af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf`;
- unique chunks / usable articles: 301 / 62,153;
- index dimensions: 62,153 × 50,000;
- official queries used to fit vocabulary: zero;
- corpus load / index build: 38.949 / 71.830 seconds.

## Regenerated V4-A-style results

Metrics aggregate three complete 299-row OOF passes (897 decisions).

| Metric | Value |
|---|---:|
| Macro F1 | 0.6813929522 |
| Label-0 F1 | 0.7066115702 |
| Label-1 F1 | 0.6561743341 |
| Accuracy | 0.6833890747 |
| Confusion matrix | `[[342,66],[218,271]]` |
| Predicted counts `[0,1]` | `[560,337]` |
| Context-present macro F1 | 0.8967656086 |
| Context-absent macro F1 | 0.4407547170 |
| Brier score | 0.1909804374 |
| Seed mean / std | 0.6813783489 / 0.0033404651 |
| Seed minimum / maximum | 0.6766693701 / 0.6840602518 |

Compared with the teammate aggregate reference, full macro F1 changed by -0.0058771986,
present-route macro F1 by -0.0279011868, and absent-route macro F1 by +0.0067834251.
Differences are expected: V8 uses strengthened repeated folds and the newly frozen fixed
0.25 retrieval cutoff, whereas the teammate summary reports a different fold graph and
fold-selected retrieval cutoffs. No parameter was changed to match the reference.

## Regenerated Candidate I results

Metrics aggregate three complete 169-row absent-route OOF passes (507 decisions).

| Metric | Value |
|---|---:|
| Macro F1 | 0.5345004669 |
| Label-0 F1 | 0.5317460317 |
| Label-1 F1 | 0.5372549020 |
| Accuracy | 0.5345167653 |
| Confusion matrix | `[[134,133],[103,137]]` |
| Predicted counts `[0,1]` | `[237,270]` |
| Brier score | 0.2806696510 |
| Seed mean / std | 0.5344246195 / 0.0100481444 |
| Seed minimum / maximum | 0.5206429247 / 0.5443148790 |

This differs from the authenticated 0.534500 reference by only +0.0000004669 and confirms
the Candidate I reproduction.

## Candidate Q-authenticated results

| Metric | Value |
|---|---:|
| Macro F1 | **0.6906556613** |
| Label-0 F1 | 0.6513409962 |
| Label-1 F1 | 0.7299703264 |
| Accuracy | 0.6956521739 |
| Confusion matrix | `[[255,153],[120,369]]` |
| Predicted counts `[0,1]` | `[375,522]` |
| Context-present macro F1 | 0.8967656086 |
| Context-absent macro F1 | 0.5345004669 |
| Seed mean / std | 0.6906323589 / 0.0052051142 |
| Seed minimum / maximum | 0.6832951789 / 0.6948149156 |
| Maximum predicted-class share | 0.5819397993 |

The result is 0.0013953387 below the 0.692051 champion. Seed stability and class-balance
guards pass, but the primary frozen score gate fails. Verdict: **rejected**. It does not
reach the 0.70 meaningful, 0.72 strong, or 0.74 high-value gates.

## OOF provenance

All 15 outer-fold records persist aggregate-safe provenance only:

- validation row indices and training/validation row counts;
- SHA-256 digests of training and validation index sequences;
- hashed training and validation group membership;
- model-configuration and corpus-manifest fingerprints;
- V4-A-style and Candidate I learned-state fingerprints;
- fold-local preprocessing fit counts and zero validation-preprocessing rows.

Every official row is validated exactly once per seed. Row overlap, group overlap,
preprocessing-scope violations, missing coverage, and fingerprint failures are all zero.
Probability orientation is explicitly label 1 and the threshold is exactly 0.50. No raw
text, vocabulary, or row-level probability is persisted.

Configuration fingerprint:
`e762ae7afb106211a1379b15a148e005df11967cc4f01706c88cade3dc24f905`.

## Runtime and ignored artifacts

- total runtime: 140.143 seconds;
- peak process working set: 2,980,638,720 bytes;
- `frozen_config.json`: 1,645 bytes,
  SHA-256 `bfe700ef09895c2ca0042c75b851147addd785e1bf58e640909aa25166f92593`;
- `oof_provenance.json`: 367,708 bytes,
  SHA-256 `dd408ef0de697dbfb330bd264b3ced70866ceb113f42962fe8bacfd8f435d4a4`;
- `results.json`: 71,683 bytes,
  SHA-256 `9be31de6abd8700e62253d36420127080c6326e74bbc20d9cc4f7df198ecdca3`.

All artifacts remain ignored. The run produced no test labels, prediction CSV, submission,
model file, checkpoint, or raw-text artifact.
