# Public Kaggle Notebook Audit

## Corrected score conclusion

The supplied Kaggle screenshots establish these public leaderboard results:

| Public notebook | Score-producing version | Verified public/best score | Audited local file |
|---|---:|---:|---|
| [olikbochon_fullsqueeze](https://www.kaggle.com/code/mahdihasanqurishi/olikbochon-fullsqueeze) | 6 | **0.684** | Version 13 |
| [notebook42f62bbcad](https://www.kaggle.com/code/nazifaanjum/notebook42f62bbcad) | 4 | **0.685** | Version 4 |
| Project BanglaBERT V3 | — | **0.525** | Exact successful project source |

Both public scores are treated as verified Kaggle results. They exceed V3 by
0.159 and 0.160 respectively. The screenshots verify scores and version
numbers; they do not resolve validation quality, data provenance, or offline
reproducibility.

The downloaded FullSqueeze notebook is Version 13, whose latest run is broken.
It is not the exact implementation that produced 0.684. Statements about the
score-producing FullSqueeze method therefore require a separate static audit
of Version 6.

## FullSqueeze Version 13 observations

- Backbone: two seeds of `csebuetnlp/banglabert_large`.
- Intended auxiliary judges: TigerLLM-9B and Qwen2.5 7B/32B checkpoints.
- External data: Bengali NLI/QA/math datasets, participant-hosted answer banks,
  and Bengali Wikipedia.
- Main likely gains: exact/fuzzy answer lookup, broad QA/NLI supervision,
  larger-backbone averaging, retrieval, and numeric-consistency signals.
- Validation weakness: the same 299 official labels drive epoch choice, feature
  gates, blend weights, and route-specific thresholds.
- Competition-specific behavior: measured labeled-sample-to-test lookup,
  participant answer banks of unresolved provenance, joint validation/test rank
  normalization, and a hard-coded test-row assumption.
- Reproducibility defects: unpinned models/datasets, undeclared offline inputs,
  ignored installation failures, and an undefined `cfg.judge_ids` reference.

These findings describe Version 13 only. They cannot be used to attribute the
0.684 score to an exact Version 6 component.

### Obtain the score-producing Version 6

This is independent of V4-A and need not block it. Either select **Version 6**
from Kaggle's version history and use **Download**, or use the version-qualified
Kaggle CLI command:

```bash
mkdir -p /tmp/olikbochon-fullsqueeze-v6
.venv/bin/kaggle kernels pull \
  mahdihasanqurishi/olikbochon-fullsqueeze/6 \
  -p /tmp/olikbochon-fullsqueeze-v6 \
  --metadata
```

Keep the download outside the repository. Before auditing, confirm the pulled
metadata says version 6 and hash the notebook. Audit source cells statically;
do not execute it or inspect its saved row-level outputs.

## Wikipedia Version 4 observations

- Model: `StandardScaler` plus class-balanced logistic regression.
- Features: context presence, token overlap, numeric support/overlap, response
  number presence, response substring, and response/context character lengths.
- Retrieval: top-one character-TF-IDF search over Bengali Wikipedia; retrieved
  snippets become context only above similarity 0.25.
- Original validation: ordinary five-fold stratified OOF on the no-context
  subset while trying multiple retrieval thresholds on the same OOF result.
- Primary likely gain: retrieval supplies relevant pseudo-context to prompts
  that otherwise have none, allowing support and numeric-consistency features
  to operate.
- Reproducibility weaknesses: a private duplicate of competition data,
  unpinned Wikipedia input, non-grouped cutoff tuning, unsafe input enumeration,
  and no strict ID/template validation.

The private competition duplicate is unnecessary. V4-A replaces it with the
official competition attachment and independently reimplements the documented
feature/retrieval method with safer validation.

## Legitimately reproducible ideas

Rank 1 is strongest in each column.

| Idea | Expected gain | Reproducibility | Offline feasibility |
|---|---:|---:|---:|
| Clean, provenance-audited QA/NLI adaptation | 1 | 4 | 3 |
| Pinned Bengali Wikipedia retrieval for no-context prompts | 2 | 2 | 2 |
| BanglaBERT-large with limited seed averaging | 3 | 3 | 4 |
| One compact authenticated local judge | 4 | 5 | 5 |
| Numeric and lexical consistency features | 5 | 1 | 1 |
| Context/no-context routing | 6 | 1 | 1 |

Do not reproduce sample-to-test label transfer, transductive test rank
normalization, hard-coded test size/order, or participant-created answer banks
until their provenance and overlap are independently cleared.

No test examples or saved notebook outputs were inspected during this audit.
