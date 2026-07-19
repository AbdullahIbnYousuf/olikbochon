# V7 Repository Audit

Audit date: 2026-07-19 (Asia/Dhaka)  
Operator: Codex  
Integration branch: `feat/v7-compatible-hybrid`

## Scope and evidence policy

This audit treats Git-tracked source, commits, preserved artifacts, and recorded experiment
outputs as execution evidence. Descriptive handoff material is treated as specification unless
the corresponding code and output can be verified. No competition test examples or row-level
test predictions were inspected or displayed.

## ZIP integrity and layout

The supplied `olikbochon_handoff.zip` is 1,042,147 bytes. It was extracted once to the ignored,
untouched directory `.handoff_original/`. Every payload SHA-256 matched `SHA256SUMS.txt`, and
`git bundle verify` reported a complete history.

The ZIP contains a Git bundle, one official labeled JSON file, four V4-A artifacts, environment
metadata, and setup scripts. It explicitly contains no competition test input, Wikipedia corpus,
model checkpoint, cache, credentials, or external public dataset.

## Git history

Git history is present. The packaged branch heads are:

- V4-A: `c79a68f522437ff849a843916f5b065d5305a9b4`
- research: `5ef12783032f8c70aeb2f5c3e595c58ce429a0de`
- merge base: `ab29d53c8530255a56637526d4edf033c0ebfe3a`

The pre-existing workspace had V4-A at `ddd40b6` and a later research remote at `48239e5`.
The packaged V4-A head is one documentation-only commit after `ddd40b6`. The new V7 branch was
created without moving the original V4-A branch, then fast-forwarded to the exact packaged head.
The research branch was not merged wholesale.

## Official labeled data

The supplied official sample has 299 rows and exact schema
`context,prompt_bn,response_bn,label`. It has no explicit row ID. Label counts are 136 for label
0 (hallucinated) and 163 for label 1 (faithful). Its SHA-256 is
`f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28`.

Observed context absence is exactly two JSON nulls plus 167 textual `[NULL]` sentinels. There
are no empty or whitespace-only contexts in the supplied sample. Correct routing is therefore
130 context-present and 169 context-absent rows. The reusable V7 policy must also safely handle
Python/NumPy missing values and conservative textual null forms without treating ordinary text
as absent.

## V4-A artifacts and probability orientation

All four preserved V4-A files passed their transferred SHA-256 checks. The OOF artifact has 299
rows with columns `row_index,fold,label,probability_label1`; the test-probability artifact has
2,516 rows with columns `id,probability_label1`; the submission has 2,516 rows and exact columns
`id,label`. `probability_label1` is explicitly the probability of faithful label 1. The old OOF
artifact is position-indexed, not stable-ID-indexed, and uses one five-fold assignment, so it is
not eligible as a common-fold E0 input or stacking feature.

The preserved summary verifies V4-A's reported nested grouped macro F1 of `0.6872701508`, route
macro F1 values of `0.9246667954` (context present) and `0.4339712919` (context absent), and the
independently supplied Kaggle public score of `0.685`. The preserved validation used 296 groups.

## Research branch audit

The audited research implementation contains working code for corrected context routing,
repeated grouped validation (seeds 17/29/43, five folds), strengthened grouping, aggregate
diagnostics, the deterministic exact-substring rule, numeric lexical features, and five sparse
context-absent candidates. Candidate I is a combined character TF-IDF plus word TF-IDF plus
numeric-feature logistic model, not the separate-field Candidate H.

The recorded strengthened graph has 291 groups. This must be regenerated from the supplied data
before V7 folds are frozen. Candidate I's reported routed macro F1 `0.692051` was produced under
that strengthened protocol and is not directly comparable to V4-A's 296-group estimate.

The branch also contains neural and later research code, but the audit does not establish
implemented atomic-claim splitting, claim-level loss, public-5K adaptation in the V7 path,
cross-fitted V7 stacking, execution profiles, or a research-branch Kaggle inference/submission
path. No such capability will be claimed without new code, tests, and artifacts.

## Available and missing resources

Available:

- complete relevant Git history;
- official 299-row labeled sample;
- V4-A OOF/test probabilities, summary, and submission;
- V4-A and research source/logs;
- project environment specifications.

Missing:

- Bengali Wikipedia dump, passage database, retrieval cache, or fitted index;
- authenticated BanglaBERT checkpoint and tokenizer;
- competition test input and sample submission input;
- public 5K data;
- local GPU visibility;
- an active project Python environment with NumPy/pandas/scikit-learn/PyTorch.

Therefore E0 retrieval regeneration, E2 semantic training, final test inference, and submission
generation are initially BLOCKED. The common-fold framework, routing/grouping, leakage tests,
E1, and any retrieval-independent fusion scaffolding remain implementable; dependency setup is
attempted separately and recorded in the environment audit.

## V4-A preservation decision

V4-A remains the verified reference. Its source branch, transferred payload, and preserved
artifacts are not modified. New outputs use `artifacts/v7/` (Git-ignored), generated notebooks
use `notebooks/generated/`, reusable code uses `src/`, and tests use `tests/`.

## Post-audit resource resolution

The initial resource blockers were later resolved with an official Wikimedia bnwiki dump and an
authenticated pinned BanglaBERT checkpoint; E0-E4 were consequently executed. The historical
Kaggle `abyaadrafid/bnwiki` corpus remained unavailable, so the replacement corpus and its
license/hash are declared explicitly and no byte-equivalence claim is made. The competition test
input remained absent, leaving only M5 execution blocked.
