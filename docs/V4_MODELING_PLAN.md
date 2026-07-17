# Version 4 Modeling Plan

## Scope and immutable safety rules

Version 4 uses only the authenticated official labeled sample. The unavailable public
20K and the locally stored public 5K are excluded from training, validation, grouping,
tokenization, model selection, and threshold selection. Competition test data is not a
development input and must not be opened before a later, separately authorized inference
gate. Leaderboard feedback cannot select any configuration.

The model is the authenticated offline snapshot
`csebuetnlp/banglabert@9ce791f330578f50da6bc52b54205166fb5d1c8c`. Loading must use
a concrete local directory, `local_files_only=True`, `trust_remote_code=False`, the V3
snapshot digest check, and the V3 label-mapping checks. No tokenizer special tokens are
added; all section markers are ordinary text.

## Frozen experiment family

The initial family has exactly four stages. A new candidate requires a new documented
experiment family created before observing its results.

1. `v3_compatible_baseline`: V3 marker-delimited prompt/context pair and response pair,
   encoded at length 256 using identical pair-level `only_first` truncation.
2. `structured_serialization`: `[QUESTION]`, `[CONTEXT]`, and `[ANSWER]` ordinary-text
   sections; null context is represented as `[NULL]`; same comparison encoder.
3. `routed_serialization`: context-present rows use question/evidence versus claim;
   null-context rows use question versus claim with no empty context block; same encoder.
4. `selected_serialization_field_aware`: the best of stages 1–3, rerun with the frozen
   field-aware token budget below.

Stages 1–3 isolate serialization under the same response-protected encoder. Stage 4 tests
the incremental value of deterministic field budgeting without opening another
serialization search.

## Field-aware truncation

Maximum encoded length is 256 tokens including pair special tokens and ordinary-text
markers. The frozen field caps are:

- prompt: 80 tokens;
- response: 96 tokens;
- context: all remaining capacity;
- context head/tail split: 50%/50% (rounding the head deterministically).

Short prompts and responses are retained in full whenever their lengths are within the
frozen caps. If either exceeds its cap, deterministic head-plus-tail retention is used.
Context receives no capacity until markers, the retained prompt, and the retained response
are accounted for. Long context uses deterministic head-plus-tail retention. The same pure
encoding function is used by later training and inference.

Only aggregate statistics may be recorded: truncation counts/percentages, mean original
and retained token lengths by field, and context-present/null-context counts. Raw text and
row-level tokenization output must not be logged.

## Repeated grouped validation

Official rows use the existing V3 grouping audit: exact normalized triples,
prompt/context families, and the predeclared character-TF-IDF near-duplicate rule. All
splits use `StratifiedGroupKFold`; there is no naive random fallback, including when every
group is a singleton.

- seeds: 17, 29, 43;
- folds per seed: 5;
- required: both labels in every validation fold;
- required: zero train/validation group intersection;
- required: every row appears in validation exactly once per seed.

If group/class feasibility fails, the run stops with the seed and fold identified. Any
fallback must be separately documented and must preserve group isolation; none is
preapproved in this family.

Each fold reports two-class macro F1, label-0 F1, label-1 F1, accuracy, confusion matrix,
context-present macro F1, and null-context macro F1. Across all folds and seeds, report
mean, population standard deviation, minimum, and maximum.

## Selection and threshold discipline

The primary candidate metric is mean ordinary two-class macro F1 at threshold 0.50.
Eligible candidates must have neither predicted class above 90%. Ties are resolved by:

1. higher label-0 F1 mean;
2. lower macro-F1 standard deviation;
3. earlier frozen candidate order.

Label-0 and label-1 F1 remain required diagnostics because the official metric wording is
ambiguous. Label-0 F1 is not the primary objective.

After the candidate is selected, its threshold grid is 0.20 through 0.80 inclusive in
steps of 0.02. Rank by macro F1, label-0 F1, proximity to 0.50, then lower threshold. A
threshold predicting either class above 90% is ineligible. Threshold selection is an OOF
tuning estimate, not an independent performance estimate.

## Frozen training configuration

- epochs: 3 fixed epochs; no post-hoc extension;
- batch size: 8;
- gradient accumulation: 2 (effective batch size 16);
- learning rate: `2e-5`;
- precision: CUDA fp16 mixed precision;
- maximum length: 256;
- checkpoint selection: validation macro F1, then lower validation loss, then earlier
  epoch;
- fold initialization: fresh authenticated base checkpoint for every fold;
- retained checkpoints: selected epoch only, under ignored `artifacts/v4/`;
- OOM policy for future training: stop and report during smoke testing; no silent
  hyperparameter change is authorized in this plan.

No training occurs in Gate 3. Smoke training and full reproduction require later gates.

## Artifact and reproducibility contract

All prospective runtime artifacts resolve under ignored `artifacts/v4/`. Gate 3 writes no
checkpoint. Experiment records must contain timestamp, Git commit, configuration, seed,
authenticated model snapshot/path role, training-row count, validation-group counts,
aggregate metrics, runtime, peak VRAM, checkpoint size, and warnings—never row text,
row-level probabilities, predictions, IDs, credentials, or private paths.

Final Kaggle inference must load a pinned local model/tokenizer and run with internet off.
No external API is permitted.
