# V9 Frozen Multilingual NLI Verification Plan

Status before execution: **frozen and authenticated**.

## Competition and data boundary

The repository rulebook permits publicly available open-weight models, pretrained-model
fine-tuning, and fully local inference. V9 uses no fine-tuning: it performs label-free frozen
inference only. The experiment uses the authenticated 299-row official sample, the 169-row
context-absent route, the frozen substring rule on the 130 context-present rows, and the
authenticated Bengali Wikipedia corpus. Competition-test rows and public labeled datasets are
out of scope.

The corpus logical manifest is
`af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf`, with 301 unique
WikiExtractor chunks and 62,153 usable articles. The retrieval index is fitted once on corpus
articles only. Four CPU threads retrieve exactly five query-centered 800-character snippets per
normalized official prompt in deterministic input order. No retrieval cutoff is selected.

## Authenticated frozen models

### Primary

- Repository: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`
- Revision: `b5113eb38ab63efdd7f280f8c144ea8b13f978ce`
- License: MIT; public and non-gated
- Architecture: `DebertaV2ForSequenceClassification`; 278,811,651 parameters
- Tokenizer: `DebertaV2TokenizerFast`; supported length 512
- Labels: entailment 0, neutral 1, contradiction 2
- Selected snapshot: 9 files, 578,287,898 bytes
- Snapshot manifest: `a6d1affda664168310a21b52e624e8f0f8b0d5b7de0bb1eb1bc957821f6062c5`
- Weight: `model.safetensors`, 557,652,046 bytes,
  SHA-256 `7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541`

### Control

- Repository: `joeddav/xlm-roberta-large-xnli`
- Revision: `b227ee8435ceadfa86dc1368a34254e2838bf242`
- License: MIT; public and non-gated
- Architecture: `XLMRobertaForSequenceClassification`; 559,893,507 parameters
- Tokenizer: native slow `XLMRobertaTokenizer` backed by SentencePiece
- Config positions: 514, supporting 512 content/special-token sequence positions
- Labels: contradiction 0, neutral 1, entailment 2
- Selected snapshot: 6 files, 2,248,900,704 bytes
- Snapshot manifest: `1a479097cc59a7cecb53f7c18e98eec6651fe8dfb422bc54b15e804ada63ac8c`
- Weight: `model.safetensors`, 2,243,825,580 bytes,
  SHA-256 `8869b0c99ad35ec8a8c92434b54383d2dfd7db8cd460e28b9944a407e3a423e4`

Both snapshots contain no executable modeling code and load with `local_files_only=True`,
`trust_remote_code=False`, and safetensors. Combined selected snapshot size is 2,827,188,602
bytes. XLM-R uses the project-local binary wheel `sentencepiece==0.2.1`; no dependency was
upgraded.

## Frozen semantic configuration

Each NLI pair uses the retrieved snippet as premise and `response_bn` as hypothesis. Maximum
length is 384 and `truncation="only_first"` preserves the hypothesis. All five passages are
scored by each model. The seven aggregates per model are maximum entailment, mean top-three
entailment, maximum contradiction, mean top-three contradiction, maximum
entailment-minus-contradiction margin, retrieval-score-weighted entailment, and the count of
entailment-dominant passages.

- Candidate R: seven mDeBERTa features plus seven frozen Candidate N retrieval aggregates.
- Candidate S: seven XLM-R features plus seven frozen Candidate N retrieval aggregates.
- Candidate T: both seven-feature NLI blocks, seven retrieval aggregates, and the 21 frozen V5
  context-absent numeric/lexical features.

Candidate T contains no sparse TF-IDF columns, Candidate I probabilities, or learned blend
weights. Its V5 rare-token map is fitted independently on each outer-training partition.

All candidates use a training-partition-only `StandardScaler` and balanced liblinear logistic
regression with `C=1.0`, `max_iter=3000`, `random_state=42`, and fixed threshold 0.50. Validation
uses strengthened group-safe five-fold splits for seeds 17, 29, and 43. No threshold, feature,
model, or retrieval setting is selected from results.

## Controlled execution

The two NLI models run sequentially on CUDA fp16. Synthetic-only batches 16, 32, and 64 are
probed, and the largest batch below 85% allocated VRAM is selected. Official inference uses two
DataLoader workers and pinned memory. After frozen features are available, 15 outer evaluations
run with at most three spawn workers and one BLAS thread each. Worker count falls to one if
system RAM is at least 80% or process RAM reaches 20 GB.

Exact command:

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

.\.venv\Scripts\python.exe -m olikbochon.v9_runner `
  --mode frozen-nli-verification `
  --corpus-path "F:\datathon\olikbochon\data\retrieval\bnwiki" `
  --primary-model-path "F:\datathon\olikbochon\data\models\v9_nli\mdeberta_xnli@b5113eb38ab63efdd7f280f8c144ea8b13f978ce" `
  --control-model-path "F:\datathon\olikbochon\data\models\v9_nli\xlmr_large_xnli@b227ee8435ceadfa86dc1368a34254e2838bf242" `
  --seeds 17 29 43 `
  --folds 5 `
  --cpu-workers 4 `
  --fold-workers 3 `
  --output-dir "F:\datathon\olikbochon\artifacts\v9\frozen_nli_verification"
```

Expected resource envelope is one to three GB RAM for the corpus/index plus one NLI model at a
time, below 85% of 12 GB GPU VRAM, about 2.83 GB immutable model storage, and approximately
10–25 minutes wall time depending on tokenizer and GPU throughput.
