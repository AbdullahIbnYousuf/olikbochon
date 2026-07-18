# V7 official-only null-route retrieval verification plan

Status: **prepared but not executed**. No full retrieval index has been built and no
candidate has been evaluated. Execution requires separate approval.

## Corpus authentication

Local source: ignored `data/retrieval/bnwiki/`, corresponding to Kaggle dataset
`abyaadrafid/bnwiki`.

The physical directory contains two byte-identical copies of every WikiExtractor chunk.
V7 hashes all candidate files, groups exact content duplicates, and deterministically uses
one canonical copy per content hash. Article text is never written to tracked logs or
documentation.

| Manifest field | Value |
|---|---:|
| Recursive files | 602 |
| Candidate `wiki_<hex>` chunk files | 602 |
| Unique chunk contents | 301 |
| Duplicate chunk files | 301 |
| Physical bytes | 625,855,930 |
| Unique logical bytes | 312,927,965 |
| Parsed strict UTF-8 JSON records | 65,256 |
| Usable articles | 62,153 |
| Duplicate URLs | 0 |
| Malformed records | 0 |
| Empty title/body/URL records | 0 |
| Too-short records | 3,103 |
| Frozen minimum normalized body length | 70 characters |
| Physical path-aware manifest SHA-256 | `ecd3f26730e951c1cac5f75931a27e6d5fbcb83d2a5ebc8cdab5692a06cfd411` |
| Logical unique-content manifest SHA-256 | `af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf` |

Every parsed record has only `id`, `url`, `title`, and `text`; label-like fields are
rejected. The logical corpus exactly matches the historical 301 chunks and 312,927,965
bytes. Its 62,153 usable articles differ from the historical 62,150 by three (0.0048%),
which is immaterial and is explained by the explicit normalized 70-character boundary.

All corpus files are covered by the repository's `data/` ignore rule, none is tracked, and
none appears in Git status.

## Fixed retrieval and snippets

The corpus-only retriever is fit once without official queries:

- query: V5-normalized official prompt;
- index documents: normalized article title plus body;
- `TfidfVectorizer` analyzer `char_wb`;
- character n-grams `(2, 4)`;
- maximum 50,000 features;
- float32 L2-normalized sparse matrix;
- deterministic ordering by descending score then hashed article identifier.

Two bounded snippets are implemented:

1. intro: first 800 normalized characters;
2. query-centered: 800-character window maximizing distinct query-token coverage, with
   earliest start as the deterministic tie-breaker.

Candidate M uses intro snippets for V4-A parity. Candidates N and P use query-centered
snippets as their predeclared primary method because localized lexical matches are the
relevant repository-supported evidence for top-k aggregation; outer validation is not
used to choose the method. No unlimited text concatenation occurs.

## Frozen candidates

### Candidate M — reproduced V4-A top-1

- retrieve top 1;
- intro snippet;
- choose acceptance cutoff inside inner grouped OOF only;
- eight features: context/evidence accepted, response-token overlap, numeric support,
  numeric overlap, response-number presence, response substring, response length, and
  bounded context length;
- standardized class-balanced logistic regression.

### Candidate N — top-5 aggregate

- retrieve exactly top 5;
- query-centered bounded snippets;
- maximum response-token coverage;
- maximum response-substring support;
- maximum numeric consistency;
- maximum character similarity;
- mean top-3 retrieval score;
- rank-1/rank-2 score gap;
- accepted-passage count;
- standardized class-balanced logistic regression.

### Candidate P — retrieval feature stacker

- Candidate M's eight evidence features from top-5 evidence;
- Candidate N's seven aggregate features;
- the frozen 21 V5 context-absent lexical numeric features;
- V5 rare-token frequencies fit on the current training partition only;
- `StandardScaler` plus `LogisticRegression(class_weight="balanced", C=1.0,
  max_iter=3000, solver="liblinear", random_state=42)`.

Candidate O/NLI is deliberately unimplemented. No NLI model repository, revision, or
download has been selected; it requires separate approval and does not block M/N/P.

## Nested strengthened validation

Outer evaluation:

- authenticated official 299-row sample only;
- neural/retrieval classifier scope: corrected 169 context-absent rows only;
- context-present predictions: frozen substring rule on 130 rows;
- seeds 17, 29, 43 and five strengthened V5 group folds per seed;
- zero exact/prompt/context/family/near-duplicate group overlap;
- both labels required in every route-specific train and validation split.

Inner selection within each outer-training null partition:

- deterministic three-fold `StratifiedGroupKFold`;
- minimum observed outer null training rows: 129;
- minimum inner training rows: 85;
- minimum inner validation rows: 42;
- both labels and zero overlap verified in all 45 inner partitions;
- retrieval cutoff grid fixed at 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40;
- classifier threshold grid fixed at 0.20 through 0.80 by 0.02;
- joint tie-break order: macro F1, label-0 F1, threshold distance from 0.50, cutoff
  distance from 0.25, lower threshold, lower cutoff;
- outer-validation rows used for selection: zero.

The TF-IDF index is unsupervised and corpus-only; official validation queries are only
transformed against the frozen index. Candidate P's lexical fitted state and every
classifier/scaler are restricted to the current inner or outer training partition.

Primary reporting uses classifier threshold 0.50 after inner-selected retrieval cutoff.
Secondary reporting uses the classifier threshold selected inside the corresponding inner
training OOF. No threshold is selected on reported outer rows.

## Frozen gates and ablations

Null-route gates: reject below 0.541763; promising at 0.58; strong at 0.62; high-value at
0.68. Routed candidates must beat 0.692051; +0.01 is meaningful; 0.70 is strong and 0.74
is high-value. Seed standard deviation above 0.04 is unstable and predicted-class share
above 90% is rejected.

Aggregate-only ablations are the frozen lexical null baseline without retrieval, M top-1,
N top-5 aggregates, and P top-5 plus lexical numeric features. Accepted and rejected/empty
retrieval subsets receive separate aggregate metrics. Candidate O is excluded.

## Safe artifacts

Execution output is confined to ignored `artifacts/v7/null_retrieval_verification/` and
contains only the corpus manifest, frozen configuration, and aggregate/fold numeric
metrics. It does not contain prompts, responses, article titles/bodies, URLs, article IDs,
evidence snippets, vocabulary terms, row-level probabilities, predictions, or submission
data. The full TF-IDF index is not persisted.

## Resource estimate

The strict normalized manifest pass took 232 seconds with allocation tracing and peaked at
261,738,092 Python-traced bytes. The historical V4-A pipeline took 5,113.5 seconds. V7
builds one corpus-only index and reuses retrieval for all nested candidates.

- expected index construction: 30–75 minutes;
- retrieval for 169 prompts using fixed top-1 and top-5: 5–15 minutes;
- 1,080 low-dimensional inner logistic fits plus outer fits: 5–20 minutes;
- expected total wall time: 60–120 minutes;
- expected peak system RAM: 8–16 GB during character TF-IDF vocabulary/matrix creation;
- corpus disk: 625.9 MB physical, 312.9 MB unique content;
- persisted V7 artifacts: expected below 5 MB because index/evidence/row values are not
  saved;
- no GPU or model-weight storage is required for M/N/P.

These are conservative estimates; the full index has intentionally not been built during
preparation.

## Proposed command — not yet authorized

```powershell
$env:PYTHONPATH = "F:\datathon\olikbochon\src"

.\.venv\Scripts\python.exe -m olikbochon.v7_runner `
  --mode null-retrieval-verification `
  --corpus-path "F:\datathon\olikbochon\data\retrieval\bnwiki" `
  --candidate-set frozen `
  --seeds 17 29 43 `
  --folds 5 `
  --output-dir "F:\datathon\olikbochon\artifacts\v7\null_retrieval_verification"
```

Running this command would build the full retrieval index and execute M/N/P. Separate
explicit approval is required.
