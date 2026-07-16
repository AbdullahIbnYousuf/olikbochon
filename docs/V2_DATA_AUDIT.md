# Version 2 Labeled-Data Audit

## Decision and source

Version 2 uses the approved competition resource identified on Kaggle as [`abidur14004/new-dataset`](https://www.kaggle.com/datasets/abidur14004/new-dataset). The prior repository gate blocked modeling because the dataset description and license metadata did not formally establish provenance or reuse terms. The current user approval authorizes its labeled files for this university event. That approval does not resolve the remaining documentation uncertainty, so this report does not describe the data as organizer-provided.

Every bundled copy of `test set.csv` is permanently excluded from training, validation, threshold selection, duplicate analysis, and local inference. No competition test text was decoded, displayed, summarized, or inspected during this audit.

## Audited labeled files

| Role | Exact filename | Rows | Label 0 | Label 1 | Context absent | Context present | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| Public development train | `bangla_hallucination_5k_train.json` | 4,000 | 2,000 | 2,000 | 1,600 | 2,400 | `0f988c5b09ba1b5214a93adc04e994a3c915fb0aff7eca9e2b1aca36aa62c07f` |
| Public threshold validation | `bangla_hallucination_5k_validation.json` | 1,000 | 500 | 500 | 400 | 600 | `308c2e55bacc7552d6421101060a9b0e8c56136dd90b5cdc55f66f6375d3fa66` |
| Public audit-only aggregate | `bangla_hallucination_5k_contrastive.json` | 5,000 | 2,500 | 2,500 | 2,000 | 3,000 | `cbba66057253545d39914c147843bdca88a299baa0ea6ae6932a09d64e2e8d13` |
| Independent official evaluation | `dataset samples.json` | 299 | 136 | 163 | 169 | 130 | `f1540e702761aa451245abb6b5dcc3934f8f3d16c5baa8851b41dbc66da24b28` |

All files have source-order columns `context,prompt_bn,response_bn,label`, integer labels, both classes, no missing labels/prompts/responses, and no `id` field. The official file has two actual context nulls; the public files have no actual nulls. Null-like context markers are normalized consistently.

## Deterministic fingerprints

The **text fingerprint** is SHA-256 over the Version 2 field-marked text after Unicode NFC, missing-context normalization, and whitespace collapse. It includes normalized prompt, context, response, and deterministic boundaries, but excludes the label. It detects text duplicates, cross-partition overlap, and identical text carrying conflicting labels.

The **labeled-row fingerprint** hashes the same normalized field-marked text plus an explicit `__LABEL__` boundary and the integer label. It detects complete labeled-row duplicates. Neither key contains raw text.

## Exact audit results

- The 4,000 and 1,000 labeled-row multisets exactly compose the 5,000 aggregate.
- Internal duplicate text rows: zero in every partition.
- Internal duplicate labeled rows: zero in every partition.
- Public-train/public-validation text overlap: zero.
- Combined-public/official text overlap: zero.
- Identical normalized text with conflicting labels: zero.
- Exact rows removed to protect holdouts: zero.
- Final exact unique count before near-duplicate handling: 5,299.

## Near-duplicate method

Only labeled data was used. Normalized field-marked text was vectorized with `char_wb` TF-IDF character 3–5 grams, sublinear TF, case preservation, `float32`, and L2 normalization. Cosine similarities were computed in blocks of 256 left-side rows. A pair was flagged only when cosine similarity was at least 0.97 and text-length ratio was at least 0.90. Reported percentiles are percentiles of each left row's best cross-partition similarity.

### Public 4k train versus public 1k validation

| Measure | Result |
|---|---:|
| Rows checked | 4,000 × 1,000 |
| High-confidence pairs | 0 |
| Affected train rows | 0 |
| Affected validation rows | 0 |
| Label agreements/conflicts | 0 / 0 |
| Maximum similarity | 0.964740 |
| Row-best p50 / p90 / p95 / p99 | 0.767696 / 0.851318 / 0.867071 / 0.913860 |

### Combined public 5k versus official sample

| Measure | Result |
|---|---:|
| Rows checked | 5,000 × 299 |
| High-confidence pairs | 0 |
| Affected public rows | 0 |
| Affected official rows | 0 |
| Label agreements/conflicts | 0 / 0 |
| Maximum similarity | 0.568418 |
| Row-best p50 / p90 / p95 / p99 | 0.113873 / 0.195352 / 0.211967 / 0.258764 |

The stop conditions did not fire. Near-duplicate removals were zero for train, validation, and official roles.

## Directory-role and test quarantine

A public-data root is any directory co-locating the exact 4k and 1k filenames. All CSV files below every such root are quarantined from test discovery. A competition root must co-locate the official `dataset samples.json`, `test set.csv`, and an organizer sample submission. Coherent byte-identical duplicate roots are resolved deterministically; non-identical roots fail. Test identity is never inferred from filename or hash alone, and the public test copy cannot be selected.
