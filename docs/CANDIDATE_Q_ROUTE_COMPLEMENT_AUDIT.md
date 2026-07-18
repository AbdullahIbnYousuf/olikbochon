# Candidate Q route-complement audit

Status: **the teammate-artifact Candidate Q remains blocked; its source-regenerated V8
replacement was evaluated and rejected**. No Candidate Q test labels, probabilities,
submission, threshold search, or leaderboard-driven decision was produced.

## Fixed candidate and acceptance rule

Candidate Q was frozen before artifact inspection:

- corrected context-present rows: V4-A label-1 OOF probability at threshold 0.50;
- corrected context-absent rows: V5 Sparse Candidate I genuine grouped OOF label-1
  probability at threshold 0.50;
- corrected official routes: 130 present / 169 absent;
- proceed only if local macro F1 exceeds 0.692051;
- at least 0.70 is meaningful.

No averaging, stacking, learned weight, threshold search, or leaderboard optimization is
allowed. The reported public scores of 0.662 and 0.685 were not used in any calculation or
decision.

## Authenticated inputs

| Input | Bytes | SHA-256 |
|---|---:|---|
| Official labeled sample | 289,118 | `f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28` |
| V4-A OOF probabilities | 8,076 | `78ee80a3f9944500090342170ccf60db7cbc793c7fbe6a17ad3e6a6906d51ea6` |
| V4-A test probabilities | 60,029 | `2dd5ffa97c285727143fefe7d172bd02588368b7c3ff111088709ce0eb27adc2` |
| V4-A run summary | 8,783 | `b30dabf16cacf0bb684791740cf1d486b73bade27ff6805edb9daf0bccffca85` |
| V5 sparse frozen config | 4,111 | `ac0dbd1debd10a150dcf0352410e416e35889ef7df658f148f57be822c504d21` |
| V5 sparse aggregate results | 607,527 | `2752bf0818cf1505e259f71ee11983a23fcdce6ede54ada61216d7e895ace1f9` |
| Champion test probabilities | 119,808 | `8ee3fb2d58e56471efe87eed8aadb20ee2f7c9d9a576d54bed71af7d810d5cbb` |
| Champion run summary | 3,983 | `9cf5b2a6b85d455eace6394d08ba1ab302def53f0bb2ac7cada0656cb52cbc94` |

No persisted Candidate I row-level OOF artifact exists. The V5 files retain aggregate,
per-seed, and fold metrics but deliberately omit row-level probabilities.

## Alignment and orientation checks

The V4-A OOF file has exactly 299 rows, 299 unique `row_index` values, and source indices
0 through 298 exactly once. Sorting by `row_index` produces zero index-alignment errors and
zero label mismatches against the authenticated official source order. Fold identifiers are
exactly 0–4. All probabilities are finite and within [0,1]. The column is explicitly named
`probability_label1`, and its labels follow the repository convention label 1 = faithful.

These checks establish row alignment and probability orientation. They do **not** establish
that each probability was produced by a model that excluded its own row.

## OOF-provenance blocker

The V4-A run summary describes an honest nested grouped estimate, and the CSV looks like a
well-formed OOF artifact. However, the preserved materials contain none of the evidence
needed to independently authenticate per-row exclusion:

- outer-training membership by fold;
- inner/outer fold manifests or group IDs;
- fitted-model or checkpoint identifiers per fold;
- training-source fingerprints tied to each fitted model;
- a reproducible V4-A generation command with retained fold outputs.

This limitation was already recorded in `docs/V4A_ARTIFACT_AUDIT.md`: V4-A training
membership cannot be authenticated from the aggregate artifacts. A fold-number column and
one value per row prove coverage, not exclusion from training.

The user-defined rule says to stop if genuine OOF status cannot be established. Therefore
Candidate I OOF regeneration was not started and Candidate Q metrics were not computed.
Reporting a score would silently relax the provenance requirement after observing the
available artifacts.

## Test-artifact structural audit

No competition-test text was opened or printed. The two approved probability artifacts
were compared structurally only:

- V4-A test rows / unique IDs: 2,516 / 2,516;
- champion test rows / unique IDs: 2,516 / 2,516;
- exact ID-order match: yes;
- champion route totals: 1,361 present / 1,155 absent.

No Candidate Q test decision, probability file, or submission was generated.

## Metrics and verdict

Candidate Q local metrics: **not available — evaluation blocked before prediction**.

The known component references remain descriptive only:

- V4-A full OOF macro F1: 0.6872701508;
- V4-A claimed context-present OOF macro F1: 0.9246667954;
- Candidate I grouped repeated-OOF null-route macro F1: 0.534500;
- current routed champion: 0.692051.

Acceptance verdict: **not evaluated; Candidate Q cannot proceed**. The 0.692051 and 0.70
gates were not applied because there is no provenance-compliant combined OOF estimate.

To unblock a future audit, regenerate V4-A predictions from retained, authenticated fold
training manifests (or rerun V4-A under a reproducible group-safe protocol) so every
official row's exclusion can be verified. This is not approval to run that experiment.

## Source-regenerated resolution

The separately approved `v8_authenticated_route_complement` experiment regenerated both
route components from source under the same strengthened seeds 17/29/43 × five folds. It
did not use the teammate V4-A OOF values for scoring. All 15 fold records authenticate
training and validation indices, hashed group membership, zero overlap, fold-local fitted
preprocessing, configuration/corpus fingerprints, and fitted-model fingerprints.

The authenticated route complement scored 0.6906556613 macro F1, below the frozen
0.692051 gate, and was rejected. Full details are in
`docs/V8_AUTHENTICATED_ROUTE_COMPLEMENT.md`. The original teammate artifact remains
inadmissible for scoring; V8 resolves the scientific question through new provenance rather
than retroactively trusting that artifact.
