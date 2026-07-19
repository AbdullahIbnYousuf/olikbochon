# Experiment Artifact Inventory

## Storage and safety

This inventory was audited on 2026-07-18 through the authenticated Kaggle CLI
2.2.3. Preserved files live below `artifacts/kaggle/`, which is Git-ignored.
Runtime CSVs, model weights, datasets, logs, and Kaggle credentials are not
tracked or staged. Checksums cover the exact local bytes downloaded from the
successful current notebook versions.

## Successful Kaggle experiments

| Experiment | Kaggle notebook and successful version | Submitted file | Public score | Current Kaggle outputs | Preservation result |
|---|---|---|---:|---|---|
| V1 | `abdullahibnyousuf/tfidf-baseline-v1`, version 2 | `submission.csv` | 0.466 | `submission.csv`, `submission_fixed_050.csv`, `submission_class0_experimental.csv` | Submitted file preserved. No run summary, OOF probabilities, test probabilities, or checkpoint exists in the current output. |
| V2 | `abdullahibnyousuf/tf-idf-baseline-v2-5299-rows`, version 2 | `submission.csv` | 0.426 | `submission.csv`, `submission_fixed_050.csv` | Submitted file preserved. No run summary, OOF probabilities, test probabilities, or checkpoint exists in the current output. |
| V3 | `abdullahibnyousuf/banglabert-v3`, version 3 | `submission.csv` | 0.525 | Final and selected Stage-A checkpoint directories, `submission.csv`, `submission_fixed_050.csv`, and `v3_run_summary.json` | Submitted file, summary, final checkpoint, and selected Stage-A checkpoint preserved. No OOF or test-probability artifact exists in the current output. The unsubmitted fixed-0.50 CSV remains available on Kaggle but was not duplicated locally. |
| V4-A | `abdullahibnyousuf/v4-a-dynamic-wikipedia-discovery`, version 2 | `submission.csv` | **0.685** | `submission.csv`, `v4a_oof_probabilities.csv`, `v4a_test_probabilities.csv`, and `v4a_run_summary.json` plus generated runtime files | All four requested experiment artifacts preserved. No model checkpoint applies to this TF-IDF/logistic-regression run. |

The competition submission history independently confirms the four scores and
successful filenames. Missing artifacts above are marked unavailable because
they are absent from the current Kaggle notebook outputs; no substitute was
fabricated. Kaggle's bulk downloader could not stream the large V3 weights
reliably, so those two authenticated output URLs were downloaded in chunks and
accepted only after both 442,499,648-byte files passed safetensors-header and
data-bound validation.

## Preserved SHA-256 checksums

| Experiment | Relative path below `artifacts/kaggle/` | Bytes | SHA-256 |
|---|---|---:|---|
| V1 | `v1_0466/submission.csv` | 16,514 | `ade53187cc76082370011d4ae1a5710b188b03a56df58efd9a6f4e1b8283ff2a` |
| V2 | `v2_0426/submission.csv` | 16,514 | `020e63fb6635f3a396ebc71925b172a7f169e711014319cbe98aed1543e0e993` |
| V3 | `v3_0525/submission.csv` | 16,514 | `151ed677ce77ba8bab3e7b459426f45c63f3fa763af04d3dccf830a219cf59e2` |
| V3 | `v3_0525/v3_run_summary.json` | 16,560 | `cab9d403dcb581441f9b3ece675ab05fb19eebdb0329e987e5876137ae805b7b` |
| V3 final | `v3_0525/banglabert_v3_model/BANGLABERT_MODEL_README.md` | 8,501 | `20ff901fcb556271eaabce5c9d868160e9fb5a52a5e11136c66adc667d710496` |
| V3 final | `v3_0525/banglabert_v3_model/BANGLABERT_NORMALIZER_NOTICE.md` | 1,280 | `83c501da369a3b3bc162437d329047e4ddebcf645dff501ca5bf93d6966346d9` |
| V3 final | `v3_0525/banglabert_v3_model/FTFY_LICENSE.txt` | 1,094 | `0228ed7b72a62934309a39f5900b4d2e693e0ead88f54e459b36328243c3be63` |
| V3 final | `v3_0525/banglabert_v3_model/SNAPSHOT_INFO.md` | 2,104 | `400abbaac9d4d496279c8e9d15719526f400f5f53d26d0fd46b5eb6d0fb6067e` |
| V3 final | `v3_0525/banglabert_v3_model/V3_ATTRIBUTION.md` | 443 | `8a21db8d330a286898890751c54cf0e0164924d9ac2aaefa25d58df84b8dc412` |
| V3 final | `v3_0525/banglabert_v3_model/V3_METADATA.json` | 880 | `15f1dec8c3ce03499347836a634ada5acd1242a35e215e469679e1a9e5d1ed8e` |
| V3 final | `v3_0525/banglabert_v3_model/WCWIDTH_LICENSE.txt` | 1,322 | `70b98a95a2144eb70af8017fa8c6d95ce247e40867436e8bc649e137fe13d21a` |
| V3 final | `v3_0525/banglabert_v3_model/config.json` | 1,015 | `ee14e86475a7be67bfbf98c49bd6eb6d3156f61aefedda4c05d199120e029ca8` |
| V3 final | `v3_0525/banglabert_v3_model/model.safetensors` | 442,499,648 | `80a8a8614d5fa5d99235c9a0242ed50ceffcf7fe349a798515faf183d3a4274c` |
| V3 final | `v3_0525/banglabert_v3_model/tokenizer.json` | 1,031,937 | `a6556d0f2a55b54b05ae404741ae679ac3d8a79b5016cdcc2037e39c5a20abe3` |
| V3 final | `v3_0525/banglabert_v3_model/tokenizer_config.json` | 382 | `0222cebdb4dd30a03464a38c24c5f944ab61cc5fd567f8383d123e0c502bfd38` |
| V3 Stage A | `v3_0525/banglabert_v3_stage_a_selected/config.json` | 1,015 | `ee14e86475a7be67bfbf98c49bd6eb6d3156f61aefedda4c05d199120e029ca8` |
| V3 Stage A | `v3_0525/banglabert_v3_stage_a_selected/model.safetensors` | 442,499,648 | `3647bcd87c5fe3475826349b481fdb550a342db5981c308a0a476df3f1d5040e` |
| V3 Stage A | `v3_0525/banglabert_v3_stage_a_selected/tokenizer.json` | 1,031,937 | `a6556d0f2a55b54b05ae404741ae679ac3d8a79b5016cdcc2037e39c5a20abe3` |
| V3 Stage A | `v3_0525/banglabert_v3_stage_a_selected/tokenizer_config.json` | 382 | `0222cebdb4dd30a03464a38c24c5f944ab61cc5fd567f8383d123e0c502bfd38` |
| V4-A | `v4a_0685/submission.csv` | 16,514 | `bdfcd907177fb35537ae33998c6427f65d3ce574d18f82ee3e656bdb4ee2fa19` |
| V4-A | `v4a_0685/v4a_oof_probabilities.csv` | 8,076 | `78ee80a3f9944500090342170ccf60db7cbc793c7fbe6a17ad3e6a6906d51ea6` |
| V4-A | `v4a_0685/v4a_run_summary.json` | 8,783 | `b30dabf16cacf0bb684791740cf1d486b73bade27ff6805edb9daf0bccffca85` |
| V4-A | `v4a_0685/v4a_test_probabilities.csv` | 60,029 | `2dd5ffa97c285727143fefe7d172bd02588368b7c3ff111088709ce0eb27adc2` |

## Runtime-source identity

The generated V4-A notebook embeds runtime archive SHA-256
`9dd66e6f70223adbaa3bd84a52f0c67d5009ea3aed8e627874cb1bd4886fc63c`.
The successful Kaggle version 2 used that same value. Repository tests parse
the committed generated notebook and rebuild the archive from canonical source
to enforce exact equality.
