# Risk Register

Ratings are qualitative for planning. A “gate” means implementation or submission must stop until the mitigation is satisfied.

| ID | Risk | Current evidence | Likelihood | Impact | Mitigation / gate |
|---|---|---|---|---|---|
| R01 | Internet dependency | Starter metadata enables internet and its NLI cell uses a remote Hugging Face ID. | High | Critical | Attach pinned local artifacts; set offline flags and `local_files_only=True`; pass a Kaggle internet-off smoke test. |
| R02 | Hugging Face download failure | No model sources are attached; `pipeline` must resolve config/tokenizer/weights. | High | High | Never resolve remote IDs in final inference; verify manifest before model construction. |
| R03 | Python/library incompatibility | Local setup uses Python 3.14; Kaggle image and model cards may use different Python/Transformers versions. | Medium | High | Test in the exact Kaggle image; record versions; avoid unsupported APIs; attach offline wheels only if necessary. |
| R04 | Kaggle GPU memory limit | Transformers can exceed 16 GB with long sequences/batches/optimizer state. | Medium | High | Start with base models, fp16/bf16 where supported, length 256, small batches, gradient accumulation; record peak VRAM. |
| R05 | Runtime over 9 hours | Unbatched NLI and excessive CV/ensembles multiply runtime. | Medium | Critical | Batch inference, benchmark full scale, cap folds/models, keep a linear/single-model fallback, require a runtime margin. |
| R06 | Label-direction mistake | Official labels are `0 = hallucinated`, `1 = faithful`; libraries often default to class 1 as positive. | Medium | Critical | Central label map; explicit metric labels/`pos_label=0`; unit tests with known synthetic arrays; verify model `classes_`. |
| R07 | Official metric ambiguity | Rulebook mixes “Macro F1” with “on ... label 0”; starter declares macro F1. | High | High | Ask organizers for exact evaluator/sklearn equivalent; report both; provisionally select on class-0 F1. |
| R08 | Train/test leakage | Public download contains a CSV with the exact official test hash. | High | Critical | Explicit labeled-file allowlist; prohibit all CSVs from training; fatal hash/path checks; never inspect test text. |
| R09 | Duplicate leakage | Public aggregate fully contains its train/validation files; synthetic near-duplicates may cross folds. | High | High | Use aggregate or subsets, never both; exact and near-duplicate grouping before splits. |
| R10 | Test-row inspection | Manual viewing/labeling is prohibited. | Low with controls | Critical | Metadata-only audit tools; no `head`, sampling, logging, plots, or error dumps containing test text. |
| R11 | Submission-format errors | Starter invents IDs if absent and does not validate exact output contract. | Medium | Critical | Fail on missing/duplicate IDs; preserve input IDs/order; exact `id,label`; integer `{0,1}`; dynamic row/ID equality tests. |
| R12 | Missing model/tokenizer files | An incomplete Kaggle attachment can load partially then fail offline. | Medium | High | SHA-256 manifest for config, tokenizer, vocab, special tokens, weights, threshold, and preprocessing; fail early. |
| R13 | Public-leaderboard overfitting | Four daily submissions and visible scores invite iterative threshold/model tuning. | Medium | High | Predeclare local validation; log rationale; select from stable folds/seeds; use leaderboard only as a coarse sanity check. |
| R14 | Incompatible model license | BanglaBERT is CC BY-NC-SA 4.0; public 5k metadata says license `unknown`. | High | Critical | Obtain written/organizer confirmation; retain license/citation files; prefer MIT/Apache models if unresolved. |
| R15 | Repository data leakage | Data, ZIPs, outputs, credentials, and weights must never enter Git. | Medium | Critical | Keep ignore rules; stage explicit docs/code paths; audit `git diff --cached --name-only` and tracked file sizes before every push. |
| R16 | Wrong public dataset version | Directory/request says 20k, but current dataset has only 5k distinct labeled rows. | High | Medium | Confirm intended Kaggle dataset/version with the team; record dataset ID/version/hash; never fabricate or silently merge duplicates. |
| R17 | Undocumented public label semantics | Two inspected examples align with official direction, but metadata is empty and no exact official overlap exists. | Medium | Critical | Confirm with owner/organizer documentation before use; treat public data as blocked meanwhile. |
| R18 | Context-regime imbalance | Many labeled records use null-like context; closed-book and grounded tasks differ. | High | High | Normalize context consistently; stratify/report by `has_context`; test specialists only with enough data. |
| R19 | Threshold overfitting | Choosing and reporting a threshold on the same OOF predictions inflates the selected score. | Medium | High | Nested or separated tuning/evaluation predictions; freeze threshold before untouched holdout. |
| R20 | Model artifact size creep | Multiple framework variants, optimizers, and ensembles can waste the 50 GB allowance. | Medium | High | Package only inference weights/tokenizers; exclude optimizer/TF/Flax/ONNX duplicates; audit total size. |
| R21 | Preprocessing drift | BanglaBERT requires a specific normalizer; train/inference versions may diverge. | Medium | High | Vendor/pin permitted normalization code, test golden synthetic strings, include hashes and license. |
| R22 | Hard-coded test assumptions | Starter has a fallback ID range; held-out rerun may differ in rows/order/IDs. | Medium | Critical | Dynamic schema/ID handling and tests with varied synthetic row counts/orders/types; never encode observed test values. |

## Questions that block higher phases

1. What exact implementation does the official evaluator use: class-0 binary F1 or two-class macro F1?
2. Is `abidur14004/new-dataset` the intended external dataset/version despite containing 5k rather than 20k unique labeled records?
3. What license and provenance authorize use of that public 5k and redistribution of models trained on it?
4. If BanglaBERT becomes a candidate, do organizers consider its CC BY-NC-SA 4.0 terms compatible with the competition and Kaggle model publication?
