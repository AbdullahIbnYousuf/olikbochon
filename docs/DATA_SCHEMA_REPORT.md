# Data Schema Report

## Safety boundary

This report was produced without displaying or semantically inspecting any competition test text. Test checks were limited to filename, size, row count, columns, inferred dtypes, and ID integrity. Labeled-data inspection was limited to schema/count diagnostics and exactly one public training example per label to investigate otherwise undocumented label direction; no example text is reproduced here.

Observed row counts are descriptive metadata only. They must never be hard-coded into inference or submission logic.

## File inventory

### `data/competition/`

| File | Size (bytes) | Role |
|---|---:|---|
| `bengali-hallucination.zip` | 615,686 | Retained official archive; never use directly in modeling. |
| `dataset samples.json` | 289,118 | Official labeled sample. |
| `sample submission.csv` | 19,029 | Official output template. |
| `test set.csv` | 2,329,947 | Official private competition test; metadata-only access. |

### `data/public-20k/`

| File | Size (bytes) | Role and warning |
|---|---:|---|
| `new-dataset.zip` | 1,843,973 | Retained public-dataset archive. Kaggle metadata says license `unknown`. |
| `bangla_hallucination_5k_contrastive.json` | 4,947,994 | 5,000-row aggregate labeled file. |
| `bangla_hallucination_5k_train.json` | 3,959,445 | 4,000-row subset of the aggregate. |
| `bangla_hallucination_5k_validation.json` | 988,551 | 1,000-row subset of the aggregate. |
| `dataset samples.json` | 289,118 | Byte-identical copy of the official competition labeled sample; do not add twice. |
| `test set.csv` | 2,329,947 | Byte-identical copy of the official competition test; quarantine completely. |

The directory name “public-20k” does not describe the current download: only 5,000 distinct public labeled text signatures were found, not 20,000.

## Labeled schemas and distributions

All JSON files are top-level lists of record objects. None contains an `id` field, so duplicate-ID checks are not applicable to labeled data.

| File | Rows | Columns in source order | Label dtype | Label 0 | Label 1 | Actual nulls by column | Exact duplicate rows |
|---|---:|---|---|---:|---:|---|---:|
| Competition `dataset samples.json` | 299 | `context`, `prompt_bn`, `response_bn`, `label` | integer (`int64` in pandas) | 136 | 163 | `context`: 2; others: 0 | 0 |
| Public `bangla_hallucination_5k_contrastive.json` | 5,000 | same | integer (`int64`) | 2,500 | 2,500 | all 0 | 0 |
| Public `bangla_hallucination_5k_train.json` | 4,000 | same | integer (`int64`) | 2,000 | 2,000 | all 0 | 0 |
| Public `bangla_hallucination_5k_validation.json` | 1,000 | same | integer (`int64`) | 500 | 500 | all 0 | 0 |
| Public `dataset samples.json` | 299 | same | integer (`int64`) | 136 | 163 | `context`: 2; others: 0 | 0 |

### Python value types and null-like context

The official sample has 297 string and 2 integer `response_bn` values, which is why pandas infers `object`. All prompts are strings. Its context has 2 actual nulls and 167 additional null-like string sentinels, for 169 context-absent records after normalization.

Each 5k-family file stores all three text fields as strings. Null-like context counts are:

- aggregate: 2,000;
- train subset: 1,600;
- validation subset: 400.

Prompt and response have no null-like values in these labeled files.

Recommended canonical preprocessing:

1. require the exact four labeled columns;
2. reject missing labels and labels outside `{0,1}`;
3. coerce prompt and response to strings without altering their content;
4. normalize actual null, empty/whitespace, `"[NULL]"`, and stringified null sentinels to empty context;
5. derive `has_context` after normalization; and
6. add a `source` column in memory for grouped validation/auditing, never as a model shortcut unless explicitly justified.

No column renaming is required: official and public labeled records already use `context`, `prompt_bn`, `response_bn`, and `label`. Reusable code should nevertheless map fields by validated names rather than position.

## Duplicate and overlap audit

- The two `dataset samples.json` files have the same SHA-256 (`f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28`) and all 299 text signatures/labels match.
- The 5,000-row contrastive file overlaps the train file on 4,000 text signatures and the validation file on 1,000. No conflicting labels were observed for those signatures.
- The 4,000 train and 1,000 validation subsets do not overlap each other.
- There is no text-signature overlap between the distinct 5,000 public records and the official 299-row sample.
- No exact duplicate row occurs within any individual labeled JSON.
- Near-duplicate semantic families were not computed in this analysis and must be checked before splitting.

Safe combination rule: use either the 5,000-row aggregate or the 4,000+1,000 split, never both. Add the official 299 rows only once. This would yield 5,299 distinct exact text signatures, but only if the public dataset passes the license and provenance gates below.

## Label-definition consistency

Authoritative sources—the official rulebook and starter notebook—define `0 = hallucinated` and `1 = faithful` for the competition sample.

The public Kaggle metadata for `abidur14004/new-dataset` has an empty description, no label documentation, and an `unknown` license. There is no exact overlap between its distinct 5,000 records and the official labeled sample, so labels cannot be cross-verified record-for-record. One public labeled training example from each class was inspected solely to resolve direction; both were clearly consistent with the official meaning, and the overlapping public train/validation/aggregate files have no label conflicts.

Conclusion: the public labels are empirically consistent with the official direction but not formally documented. The prior use gate was unresolved. For Version 2, the user approved the labeled files for this university event under dataset identifier `abidur14004/new-dataset`. This approval permits the scoped experiment but does not resolve the remaining provenance, citation, or unknown-license documentation uncertainty.

## Version 2 near-duplicate update

Version 2 rechecked exact fingerprints and added a deterministic cross-partition `char_wb` TF-IDF 3–5 gram audit with cosine threshold 0.97 and length-ratio threshold 0.90. Public train versus validation produced zero flagged pairs (maximum 0.964740). Combined public versus official produced zero flagged pairs (maximum 0.568418). No label conflicts or holdout-protection removals occurred. Full aggregate-only results are recorded in `docs/V2_DATA_AUDIT.md`.

## Competition test metadata only

| Property | Observed value |
|---|---|
| Filename | `data/competition/test set.csv` |
| Size | 2,329,947 bytes |
| Rows | 2,516 |
| Columns | `id`, `context`, `prompt_bn`, `response_bn` |
| Inferred dtypes | `id`: integer; all text columns: string |
| Missing IDs | 0 |
| Duplicate IDs | 0 |

No test prompt, context, response, or example was printed, summarized, or inspected.

The `data/public-20k/test set.csv` file has the same SHA-256 as the official test (`db75049956c6fa00e4d9c476716ee34bc4cc17a737f52ada06d2c0f80d567b81`). Its placement in an external/public download is a serious leakage risk. Exclude it by explicit allowlist, not merely by filename conventions.

## Sample submission structure

The official sample has 2,516 rows and exactly `id,label` in that order. Both columns infer as integers; IDs have no missing values or duplicates. Its label values are placeholders and must not be treated as evidence. Final validation must compare IDs to the input test IDs without assuming the observed row count, order, or dtype in future held-out reruns.

## Data-use decision

| Source | Decision now | Reason |
|---|---|---|
| Official competition sample | Approved for development | Official, documented labels. |
| Public 5k aggregate | Audit-only in Version 2 | User-approved for this university event, but the aggregate duplicates the selected 4k/1k roles and must never be fitted alongside them; provenance/license documentation remains incomplete. |
| Public 4k/1k split | Approved for Version 2 | User-approved for this university event. Use 4k for development and 1k for threshold selection; cite `abidur14004/new-dataset`; retain the provenance/license caveat. |
| Public copy of official sample | Exclude | Exact duplicate. |
| Public copy of competition test | Permanently exclude | Exact official-test duplicate; leakage and rules risk. |
