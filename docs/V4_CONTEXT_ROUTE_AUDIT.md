# V4 Official-Sample Context Route Audit

## Scope

This read-only audit used only the authenticated 299-row official labeled sample at
`data/competition/dataset samples.json`. It did not read competition-test rows, print or
persist row-level values, or run training.

## Schema and sentinels

The top-level JSON value is a list of 299 records. Every record has exactly four fields:
`context`, `prompt_bn`, `response_bn`, and `label`. In raw JSON, `context` contains 297
strings and two Python null values; `prompt_bn` contains 299 strings; `response_bn`
contains 297 strings and two integers; and `label` contains 299 integers. Missing-value
counts are two for `context` and zero for every other field.

Context absence representations are aggregate counts only:

| Representation | Count |
|---|---:|
| JSON null / Python `None` | 2 |
| NaN in the loaded pandas frame | 2 |
| Empty string | 0 |
| Whitespace-only string | 0 |
| Literal `[NULL]` | 167 |
| Literal `NULL` | 0 |
| Trimmed/case-normalized `[NULL]` | 167 |
| Trimmed/case-normalized `NULL` | 0 |
| Trimmed/case-normalized `NONE` | 0 |
| Trimmed/case-normalized `N/A` | 0 |
| Other recognized sentinel-like values | 0 |

The two JSON nulls become pandas NaN values during frame construction; these are the same
two rows and must not be added together.

## Independent route definitions

| Definition | Present | Absent | Present label 0 / 1 | Absent label 0 / 1 |
|---|---:|---:|---:|---:|
| A: Python null, NaN, empty, whitespace | 297 | 2 | 135 / 162 | 1 / 1 |
| B: A plus `[NULL]`, `NULL`, `NONE`, `N/A` after normalization | 130 | 169 | 47 / 83 | 89 / 80 |
| Legacy route used by completed runs | 297 | 2 | 135 / 162 | 1 / 1 |
| Corrected V4 route | 130 | 169 | 47 / 83 | 89 / 80 |

## Root cause and experiment validity

The historical V3 presence function rejects only Python missing values and blank strings.
Because `[NULL]` is a nonempty string, it is truthy. V4 originally reused that function,
then normalized the context after routing; the audited normalizer preserves `[NULL]`.
There is only one context-related field, so no alias or column confusion occurred.

The external 167/132 report detected the 167 string sentinels but omitted the two actual
nulls. The schema-correct result is therefore 169 absent and 130 present.

Completed losses, predictions, overall metrics, probability diagnostics, and exact reload
checks remain valid descriptions of the legacy-compatible pipeline that ran. They are not
valid measurements of corrected V4 preprocessing. Route-specific metrics and route-count
truncation summaries are invalid. Group partitioning and all 15 fold index assignments
were independently confirmed unchanged by the corrected route policy.

V3 behavior remains unchanged for historical reproducibility. V4 now detects official
sentinels before serialization and uses the same policy for route reporting and grouped
text construction. Historical-control training remains paused.
