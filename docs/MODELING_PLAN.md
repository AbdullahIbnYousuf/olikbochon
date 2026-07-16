# Modeling Plan

## Decision principles

1. The official rulebook and official Kaggle competition files are the source of truth.
2. Start with the smallest deterministic offline system that exercises the entire train/validate/package/infer contract.
3. Do not use the external public data until its license, provenance, citation, and intended 5k-versus-20k version are confirmed.
4. Never read competition test text for analysis; inference code may transform it mechanically but must not display it.
5. Measure class-0 F1, class-1 F1, macro F1, accuracy, and context-subset performance. Do not optimize accuracy.
6. Every transformer must be attached as a complete pinned Kaggle Dataset/Model and load with internet disabled.

## Metric interpretation gate

The official rulebook says “Macro F1 on the HALLUCINATED class (label = 0).” These are two different standard definitions:

- class-0 binary F1: `f1_score(y_true, y_pred, pos_label=0, average="binary")`;
- macro F1: the unweighted mean of class-0 F1 and class-1 F1.

The starter notebook calls the second quantity primary. The observed class-0-only threshold search collapsed toward an all-zero predictor: all-zero already scores about 0.6253 class-0 F1 on the current distribution, while the selected 0.71 threshold achieved only a tiny class-0 gain and destroyed class-1 F1. Until organizers clarify the evaluator, `macro_f1_oof` is therefore the provisional deployment default. Keep `fixed_050` as a clean reference and `class0_f1_oof_experimental` only as a diagnostic. Threshold choices must be revisited after clarification; public leaderboard probing is prohibited and is not a substitute for clarification.

## Phase A — Reliable offline baseline

### A1. Canonical data contract

- Require labeled columns `context,prompt_bn,response_bn,label`.
- Require inference columns `id,context,prompt_bn,response_bn`.
- Normalize null-like context to empty text and derive `has_context`.
- Coerce prompt and response to strings; preserve the lexical content and Bengali digits.
- Serialize fields with explicit markers such as `PROMPT`, `CONTEXT`, and `RESPONSE`, so boundaries remain visible.
- Build a source allowlist. Never glob all JSON/CSV files from either data directory.
- Initially use only the official 299-row labeled sample. Admit one copy of the public 5k only after its data-use gate passes.

### A2. Feature families

Build sparse features with a `FeatureUnion`/`ColumnTransformer`:

- word TF-IDF, 1–2 grams, on field-marked text;
- character `char_wb` TF-IDF, approximately 3–5 grams, which is robust to Bengali morphology, spacing, and spelling variants;
- separate context-response character/word branches if memory allows;
- small deterministic numeric features: `has_context`, text lengths, prompt-response/context-response token overlap, digit-set agreement, and obvious named-entity overlap counts.

Fit vectorizers on training folds only. Start with conservative `min_df` and feature caps; record every setting in a config.

Version 1 locks the configuration rather than tuning it. The word branch uses 1–2 grams, `max_features=20_000`, `min_df=1`, `max_df=1.0`, `sublinear_tf=True`, `lowercase=False`, `strip_accents=None`, `dtype=float32`, token pattern `(?u)\b\w+\b`, and L2 normalization. The `char_wb` branch uses 3–5 grams, `max_features=30_000`, and the same document-frequency, case, accent, dtype, sublinear-TF, and normalization settings. `min_df=1` is intentional because the official labeled sample is small and rare Bengali terms may be informative.

### A3. Linear models

Primary baseline: balanced logistic regression. It supplies probabilities for threshold tuning and later ensembling.

Version 1 fixes logistic regression at `C=1.0`, `solver="liblinear"`, `max_iter=2000`, `class_weight="balanced"`, and `random_state=42`. These values are not tuned.

One challenger: class-weighted linear SVM. Tune its decision threshold directly; use probability calibration only if calibrated probabilities materially help an ensemble. Calibration must be fitted inside training folds, never on the evaluation fold.

### A4. Validation and thresholding

- Use stratified, duplicate-aware folds as defined in `VALIDATION_PLAN.md`.
- Compute class-0 F1, class-1 F1, macro F1, confusion matrix, and accuracy.
- Support three named strategies: `fixed_050`, provisional-default `macro_f1_oof`, and nondefault `class0_f1_oof_experimental`. Verify classifier `classes_` before selecting the correct probability column.
- Report the default-threshold score beside the tuned score so threshold gains are visible.
- Keep threshold selection independent from any competition-test or leaderboard observations.
- Use nested 5×3 CV for separate honest macro-F1-selected and experimental class-0-F1-selected estimates. Select each threshold only from inner OOF predictions, then apply it to the untouched outer fold.
- Separately select both thresholds from standard full 5-fold OOF predictions and label both scores optimistic tuning estimates. Macro selection ranks by macro F1, class-0 F1, proximity to 0.50, then lower threshold; experimental class-0 selection reverses the first two criteria.
- Report all-zero, all-one, and majority-class diagnostics. Warn when either predicted class exceeds 90%, because high class-specific F1 can reflect class collapse rather than useful discrimination.

### A5. Submission guardrail

The later inference implementation must:

1. require nonmissing test IDs while preserving any organizer-provided duplicates;
2. preserve every ID exactly and in input order;
3. produce integer labels in `{0,1}`;
4. produce exactly `id,label` in that order;
5. require output length and ID sequence to match the input dynamically;
6. fail rather than synthesize IDs; and
7. write only to ignored output storage or `/kaggle/working/`.

Phase A is expected to run on CPU in minutes and requires no internet or model download.

## Phase B — Strong transformer baselines

Do not attempt all three immediately. Fine-tune one after Phase A is stable, then add another only if validation shows complementary value. GPU/time figures below are planning estimates for roughly 5k labeled examples, sequence length 256, mixed precision, 2–4 epochs, and a 16 GB T4; benchmark them in Kaggle before relying on them.

| Model | Why it may work | Size and GPU estimate | Time estimate | Language support | Offline attachment | License/access | Beginner risk |
|---|---|---|---|---|---|---|---|
| [`csebuetnlp/banglabert`](https://huggingface.co/csebuetnlp/banglabert) | Bengali-native ELECTRA discriminator; its model card reports 110M parameters and strong Bengali classification/NLI results. | 443 MB checkpoint; about 4–8 GB training and 1–2 GB inference VRAM at length 256, depending on batch/optimizer. | Roughly 10–40 minutes per run; normalization adds preprocessing complexity. | Bengali native. | Attach checkpoint, tokenizer, config, pinned normalization code, and classification head; load from a local Kaggle path. | Model card declares CC BY-NC-SA 4.0. Noncommercial/share-alike terms and redistribution of fine-tuned weights need explicit competition/legal confirmation. | Medium technically, **high license risk**; older model-card example targets Transformers 4.11 and requires a special normalizer. |
| [`FacebookAI/xlm-roberta-base`](https://huggingface.co/FacebookAI/xlm-roberta-base) | Robust multilingual encoder; Bengali is represented within its 94-language pretraining and it is a common cross-lingual classification baseline. | About 270M parameters; 1.12 GB safetensors. Plan for about 7–12 GB training and 2–4 GB inference VRAM. | Roughly 20–70 minutes per run. | Multilingual rather than Bengali-specific. | Attach only the pinned PyTorch/safetensors snapshot, tokenizer, config, and final head; avoid the repository’s redundant TF/Flax/ONNX files. | MIT model-card license; public and ungated. | Low-to-medium; safest general transformer choice but heavier than BanglaBERT. |
| [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased) | BERT-base architecture pretrained for 17 Indian languages and transliterations, explicitly including Bengali; useful if code-mixed/transliterated language matters. | Approximately 237M parameters from its config; 953 MB PyTorch checkpoint. Plan for about 7–12 GB training and 2–4 GB inference VRAM. | Roughly 20–70 minutes per run. | Bengali is native to the 17-language pretraining mix; also trained on transliterations. | Attach a pinned local snapshot and fine-tuned classification head; load with `local_files_only=True`. | Apache-2.0 model-card license; public and ungated. | Medium; large vocabulary increases memory, and performance must be validated against XLM-R. |

All three are far below the 50 GB limit individually. Do not package unused framework variants or optimizer states in the inference dataset. Prefer safe tensor weights when the upstream model provides them.

Recommended transformer order after Version 1: XLM-R base first for licensing and ecosystem simplicity; MuRIL second if transliteration/context results suggest value; BanglaBERT only after license and normalizer packaging are resolved.

## Phase C — Context-aware improvements

### Explicit formatting

Use stable separators and truncation rules. Preserve response and prompt, then allocate the remaining token budget to context rather than blindly truncating the concatenated string. Log only lengths, never test text.

### Separate context regimes

Report and optionally model `has_context=True` and `False` separately. Options, in increasing complexity:

1. one classifier with a `has_context` indicator;
2. one shared encoder with regime-aware features;
3. two specialist classifiers blended by the deterministic context flag, only if both subsets have enough labeled data.

Closed-book rows cannot be fact-checked with external APIs. Their signal must come from permitted training data and local models.

### Direct NLI-style classification

For context-present rows, encode `(context, response)` as a sequence pair and use entailment/neutral/contradiction probabilities as features or train the competition label directly. A direct `AutoModelForSequenceClassification` path is preferable to treating a full response as a zero-shot label. Batch inputs, set truncation explicitly, and attach every model file locally.

### Consistency features

Add deterministic features cautiously:

- numeral/date agreement between context and response;
- named-entity string overlap and unmatched-entity counts;
- negation markers;
- prompt-response and context-response token/character overlap;
- response length and context coverage.

Fit any learned extractor on training folds only. These features detect inconsistency but cannot prove factual correctness for context-absent items.

### Ensembling and calibration

- Blend OOF probabilities from the linear and transformer systems only after each is independently validated.
- Learn blend weights and thresholds on validation predictions, not public leaderboard scores.
- Consider simple rank/probability averaging before stacking.
- If probabilities are poorly calibrated, use fold-safe Platt or isotonic calibration; with small data, Platt scaling is less prone to overfit.
- Retain a single-model fallback that meets runtime even if the ensemble is abandoned.

## Phase D — Reproducible Kaggle packaging

### Artifact layout

Create one versioned Kaggle Dataset/Model per selected encoder containing:

- model config;
- tokenizer config, vocabulary/SentencePiece files, and special-token maps;
- fine-tuned weights without optimizer states;
- preprocessing/normalization code and its license;
- decision threshold and label map;
- training config and source-data declaration;
- SHA-256 manifest and upstream model revision/license files; and
- optional offline wheelhouse only if the Kaggle image lacks compatible libraries.

### Offline loading

- Attach artifacts through Kaggle inputs.
- Disable notebook internet.
- Set Transformers/Hugging Face offline environment variables.
- Resolve one explicit mounted directory; never call a remote fallback.
- Load with `local_files_only=True` and fail early if any manifest entry is missing or hash-mismatched.
- Pin a tested Kaggle image/library combination in documentation.

### Inference notebook

The notebook should be linear and restart-safe:

1. imports and environment/version report;
2. input/artifact discovery and manifest verification;
3. schema validation without displaying rows;
4. deterministic preprocessing;
5. batched inference under `torch.inference_mode()` for transformers;
6. threshold application with explicit label direction;
7. exact submission validation; and
8. write the macro-F1 default to `/kaggle/working/submission.csv`, the fixed reference to `/kaggle/working/submission_fixed_050.csv`, and a prominently warned experimental class-0 file to `/kaggle/working/submission_class0_experimental.csv`.

It must make no API/network calls, no assumptions about row count/order/IDs, and no display of test text. Run a full internet-disabled Kaggle smoke test and record runtime, peak VRAM, artifact size, and output checks before final submission.

## Immediate Version 1 implementation — exactly one recommendation

Implement a **balanced logistic-regression classifier over combined word and character TF-IDF features**.

- **Training data:** the official `dataset samples.json` only until the public-data license/version questions are resolved.
- **Input fields:** `prompt_bn`, normalized `context`, `response_bn`, plus `has_context`; encode explicit field markers.
- **Preprocessing:** validate schema/labels; coerce prompt and response to strings; normalize null-like context; no stemming, translation, external lookup, or learned preprocessing outside folds.
- **Features:** word 1–2 grams capped at 20,000 and `char_wb` 3–5 grams capped at 30,000, both with the locked settings above and `min_df=1`. Combine them with `FeatureUnion`; do not tune them.
- **Classifier:** logistic regression with `C=1.0`, `solver="liblinear"`, `max_iter=2000`, balanced class weights, and seed 42.
- **Validation split:** standard 5-fold stratified OOF at threshold 0.50 plus separate nested 5×3 CV estimates for macro-F1 and experimental class-0-F1 threshold selection. Select both full-OOF thresholds and clearly label both same-OOF scores optimistic tuning estimates.
- **Decision metric:** ordinary two-class macro F1 is the provisional default pending organizer clarification. Preserve class-0 F1 selection only as experimental; always report both class F1 values, macro F1, accuracy, confusion matrix, collapse warnings, trivial predictors, and context-regime scores.
- **Expected files:** reusable modules under `src/olikbochon/`; synthetic tests under `tests/`; canonical Jupytext source plus a clean generated notebook under `notebooks/generated/`; `docs/BASELINE_V1_RESULTS.md`; and `docs/KAGGLE_RUN_V1.md`. The notebook source is canonical and the `.ipynb` is generated from it.
- **Expected runtime:** under 5 minutes on a normal CPU for the current official sample; comfortably under Kaggle limits.
- **Success criteria:** deterministic reruns; no fold or convergence failures; explicit comparison against constant-label and majority baselines; no silent predicted-class collapse; stable fold scores without obvious leakage; all three exact validated `id,label` variants in synthetic fixtures; zero network/model-download calls; and all tests plus setup verification passing. Kaggle generation is now authorized only through the guarded `/kaggle/input` path; local execution must never open the real test CSV.
