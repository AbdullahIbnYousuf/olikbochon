# Starter Notebook Review

## Scope and method

This is a static review of `notebooks/original/starter-notebook-datathon.ipynb` at SHA-256 `e30851febd29c6ec4bd91f646447f54dffd573ecac44bd7775c4fc752313fc68`. The notebook was parsed as nbformat 4.4 and its code and Markdown sources were inspected. It was not executed or modified. It contains 22 cells: 8 Markdown cells and 14 code cells. No cells have execution counts and no outputs are stored, so the earlier traceback is not available for direct inspection.

The official rulebook and official Kaggle competition files take precedence over claims in the starter notebook.

## Cell-by-cell map

| Cell | Type | Purpose | Review |
|---:|---|---|---|
| 0 | Markdown | Title, labels, five-step outline | Correctly states `0 = hallucinated`, `1 = faithful`; refers to an unavailable `dataset_description.md`. |
| 1 | Markdown | “Load the data” heading | Keep. |
| 2 | Code | Imports, seed, competition sample path | Imports all baseline dependencies and points to `dataset samples.json`; no version pinning. |
| 3 | Code | Loads JSON into pandas and previews it | Loading is useful; `df.head()` is unnecessary for reproducible training and should not be retained in final inference. |
| 4 | Code | Coerces prompt/response to strings; normalizes context; creates `has_context` | Worth keeping after moving to tested reusable functions. It correctly anticipates numeric responses and null-like context values. |
| 5 | Markdown | “Explore the data” heading | Keep only in a development notebook, not final inference. |
| 6 | Code | Label and context balance summaries | Useful labeled-training diagnostics; not needed in final inference. |
| 7 | Code | Text-length histograms by label | Development-only EDA; requires Matplotlib. |
| 8 | Markdown | Describes a character TF-IDF baseline | Clear, but explicitly omits context. |
| 9 | Code | Per-field character TF-IDF branches plus balanced logistic regression | A good offline baseline skeleton. Needs context, word features, schema validation, and serializable/tested selectors. |
| 10 | Markdown | “Cross-validated evaluation” heading | Keep. |
| 11 | Code | Metric report, 5-fold stratified CV, OOF predictions/probabilities | Good starting structure, but it fits the whole pipeline twice and incorrectly declares macro F1 unambiguously primary despite the rulebook’s mixed wording. |
| 12 | Code | Confusion matrix | Keep. |
| 13 | Markdown | Submission section | Correctly identifies `id,label`; refers again to missing documentation. |
| 14 | Code | Fits all data, loads test CSV, predicts, writes `submission.csv` | Replace with guarded inference. It silently invents IDs if `id` is missing and lacks exact schema/order/label validation. Do not run during analysis. |
| 15 | Markdown | Improvement ideas: context, NLI, embeddings, subset analysis | Useful research directions. It references “section 6,” but no section-6 heading exists. |
| 16 | Code | Imports PyTorch/Transformers; defines the remote NLI model | Introduces heavyweight undeclared dependencies and a remote model identifier. |
| 17 | Code | Constructs a zero-shot pipeline | This is the offline-breaking Hugging Face code path. |
| 18 | Code | Scores context/response pairs with the NLI pipeline | Methodologically approximate and potentially slow; lacks batching and explicit truncation. |
| 19 | Code | Evaluates NLI scores at threshold 0.5 | A fixed threshold on the same available labeled subset is not a reliable validation design. |
| 20 | Markdown | Repeats a shorter improvement list | Duplicates cell 15 and should be consolidated. |
| 21 | Code | Empty cell | Remove from generated replacements. |

## Inputs and schemas expected by the notebook

| Reference | Expected location | Expected columns/contents | Status |
|---|---|---|---|
| Training sample | `/kaggle/input/competitions/bengali-hallucination/dataset samples.json` | `context`, `prompt_bn`, `response_bn`, `label` | Present locally under `data/competition/`; path must be resolved from Kaggle inputs rather than assumed. |
| Test data | `/kaggle/input/competitions/bengali-hallucination/test set.csv` | `id`, `context`, `prompt_bn`, `response_bn` | Present locally; final code must validate, never inspect, its text fields. |
| Dataset description | `dataset_description.md` | Column and format documentation | Referenced twice but not included in the pulled notebook, its metadata, or current project. |
| Hugging Face NLI snapshot | Remote ID only | Config, tokenizer, vocabulary, and model weights | Not attached as a Kaggle model/dataset; unavailable when internet is disabled. |

The training JSON has no `id`. The observed official test and sample-submission IDs are integer typed, but inference must preserve whatever valid IDs are supplied instead of assuming a type or synthesizing replacements.

## Models attempted

### Character TF-IDF plus logistic regression

- Two independent `char_wb` TF-IDF branches: one for `prompt_bn`, one for `response_bn`.
- Character n-grams of length 2–4, capped at 20,000 features per branch.
- `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`.
- Ignores `context` and `has_context`.
- CPU-compatible and requires no model download.

This is the part to retain conceptually for Version 1.

### Multilingual NLI model

- Task: `zero-shot-classification` through `transformers.pipeline`.
- Model: [`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).
- Intended use: context as premise and response as a dynamically supplied candidate label/hypothesis.
- Device: first CUDA device if available, otherwise CPU; float16 on CUDA and float32 on CPU.

The model card describes a multilingual NLI model covering Bengali and shows the direct premise/hypothesis interface. The notebook instead routes full responses through the zero-shot label API with `multi_label=True`. That pipeline normalizes entailment against contradiction for each candidate and does not expose a normal three-way entailment/neutral/contradiction probability. Direct sequence-pair classification would be clearer and less error-prone if this experiment is retained later.

## Packages and external dependencies

There are no package-install cells. The notebook assumes these are preinstalled:

| Dependency | Used for | Offline/reproducibility concern |
|---|---|---|
| Python `json` | JSON loading | Standard library; low risk. |
| NumPy | Missing NLI scores | Version unpinned. |
| pandas | Tables, cleaning, CSV/JSON handling | Version unpinned; dtype behavior changes across releases. |
| Matplotlib | EDA histograms | Not required for training/inference; may be absent locally. |
| scikit-learn | TF-IDF, logistic regression, CV, metrics | Version unpinned; solver/default and pandas-output behavior can change. |
| PyTorch | Device selection and dtypes | Heavy; assumed present. Not installed in the local planning environment. |
| Transformers | `pipeline` and model loading | Heavy; assumed present and version-sensitive. Not installed locally. |
| tqdm | NLI progress bar | Small but undeclared. |
| Hugging Face Hub/network | Resolves the NLI config, tokenizer, and weights | Prohibited dependency for final internet-disabled inference unless all files are attached locally. |

The pulled Kaggle metadata has `enable_internet: true`, no `dataset_sources`, and no `model_sources`. This is incompatible with the final offline requirement.

## Root cause of the Hugging Face failure

`NLI_MODEL` is a Hub repository name, not a local path. Calling `pipeline("zero-shot-classification", model=NLI_MODEL, ...)` internally invokes Hugging Face `from_pretrained` logic for the model config, tokenizer, vocabulary, and weights. The notebook attaches none of those files. On a fresh Kaggle runtime with internet disabled and an empty Hugging Face cache, resolution must contact `huggingface.co`; it therefore fails with a connectivity/cache `OSError` or Hub error before the model can be constructed.

The root cause is not the `zero-shot-classification` task name itself. It is the combination of:

1. a remote-only `NLI_MODEL` identifier;
2. no attached model snapshot;
3. no verified Transformers/PyTorch version bundle; and
4. internet-disabled execution.

A compliant future experiment must first obtain and license-check the complete snapshot outside final inference, publish/attach it as a permitted Kaggle Dataset or Kaggle Model, load from the mounted local directory, set offline flags, and use `local_files_only=True`. The final notebook should fail early with a clear missing-file message rather than fall back to the network.

## Likely runtime and correctness failures

1. `matplotlib`, `torch`, `transformers`, or `tqdm` may not exist in the target image; no dependency check or pinned environment is supplied.
2. The two hard-coded `/kaggle/input/competitions/...` paths may not match the actual mount layout in a copied notebook.
3. The NLI pipeline fails offline because its snapshot is not attached.
4. Model/tokenizer library compatibility is unpinned; the BanglaBERT-era and pipeline APIs can differ across Kaggle images.
5. NLI calls are unbatched and give no explicit truncation/max length. Long context-response pairs can exceed model limits or make runtime unnecessarily large.
6. The zero-shot wrapper uses an entire response as a candidate label and `hypothesis_template="{}"`; this is an opaque approximation to direct NLI and discards the neutral probability under `multi_label=True` scoring.
7. `cross_val_predict` is called twice, fitting TF-IDF/logistic regression ten times rather than five.
8. Default prediction thresholds are used even though the target metric emphasizes class 0.
9. The context column is cleaned but ignored by the baseline features.
10. The submission path invents sequential IDs when `id` is absent. This violates the no-hardcoding/no-row-assumption rule; missing IDs must be fatal.
11. Submission creation does not assert exact column order, unique/nonmissing IDs, row correspondence, integer labels, or the label domain `{0,1}`.
12. The notebook fits on all labeled data before verifying that the test input and output path are valid.
13. A `FunctionTransformer` containing a lambda is less robust for serialization and multiprocessing than a top-level named transformer.
14. Logistic regression can still emit a convergence warning; the notebook does not check it.

## Reproducibility issues

- No package versions, environment image contract, or dependency lock is recorded in the notebook.
- No input hashes or source-version manifest is checked.
- Hard-coded paths replace deterministic input discovery and validation.
- The Hugging Face revision is not pinned to a commit.
- Internet is enabled in metadata and implicitly required by the NLI model.
- The baseline is refit separately for labels and probabilities; reproducibility depends on both fits remaining identical.
- Only one CV seed is used and there is no untouched holdout.
- No threshold-selection procedure or saved threshold exists.
- No trained artifact, feature vocabulary, config, or inference contract is saved.
- No final offline smoke test is defined.

## Leakage and competition-rule risks

- Ordinary stratified CV does not group exact or near-duplicate prompt/context/response families. Synthetic contrastive variants can cross folds and inflate validation scores.
- The public download contains train/validation subsets duplicated inside a 5,000-row aggregate file. Combining all files would duplicate every public record.
- The public download also contains a CSV byte-identical to the official competition test. It must never enter training, validation, EDA, deduplication, vocabulary fitting, or feature engineering.
- Threshold selection and reporting on the same OOF predictions can give an optimistic selected score unless clearly treated as development tuning or evaluated on a separate holdout.
- The fixed NLI threshold is evaluated on the same labeled subset used to motivate it.
- Enabling internet and downloading the model during final inference breaks the offline rule.
- `submission.head()` would reveal IDs and generated labels; it is unnecessary in final inference, though it does not display test text.
- The fallback-generated ID sequence violates held-out rerun requirements.

## What to keep

- Explicit label direction.
- JSON loading into a DataFrame after schema validation.
- String coercion for prompt and response.
- Central context normalization and a `has_context` feature.
- Character TF-IDF as an offline Bengali-friendly baseline.
- Balanced linear classification as a baseline option.
- Stratified evaluation, class-specific F1 values, macro F1, and confusion matrix.
- The idea of reporting context-present and context-absent subsets separately.

## What to replace

- Replace notebook-global preprocessing with reusable, tested functions in `src/`.
- Replace the prompt/response-only features with field-marked prompt, context, and response word/character features.
- Replace duplicated CV fits with one probability-producing OOF pass.
- Replace the ambiguous primary-metric statement with all required metrics and an explicit organizer-clarification gate.
- Replace default/fixed thresholds with validation-only threshold selection.
- Replace ordinary CV with duplicate-aware stratified grouping where possible.
- Replace the NLI zero-shot wrapper with a direct, batched sequence-pair classifier only after an offline snapshot is attached and benchmarked.
- Replace silent ID fallback and loose CSV writing with a strict submission validator.
- Replace remote model names with pinned local paths and an artifact manifest.
- Consolidate the duplicated improvement sections and remove the empty cell.
