# BanglaBERT Version 3 Plan

## Current status

The complete self-contained Version 3 Kaggle implementation is ready for a
fresh GPU run, but no Version 3 training has been executed yet.

- The official BanglaBERT snapshot is pinned, authenticated, private on Kaggle,
  and verified for local-only loading.
- The tokenizer, pretraining architecture, and downstream classification
  transition pass CPU-only synthetic smoke tests.
- The official normalizer is the pinned behavioral reference. The Kaggle
  runtime bundles a default-only stdlib implementation plus authenticated
  `ftfy==6.0.3` and `wcwidth==0.8.2` source, with 79-case exact parity and a
  frozen startup digest gate.
- Deterministic field normalization, context presence, pair construction,
  faithful-logit resolution, vocabulary validation, and response fallback are
  implemented with synthetic unit tests.

These sources supersede every older plan reference to `reasat/banglabert`.
Only `data/models/banglabert-official-9ce791f` locally and private Kaggle input
`abdullahibnyousuf/banglabert-official-snapshot-9ce791f` are approved.

## Locked preprocessing contract

1. Determine context presence from the raw scalar. Null, `None`, NaN, empty,
   and whitespace-only values are absent.
2. Convert absent context to empty text.
3. Normalize prompt, context, and response separately with the pinned official
   NFKC-default normalizer.
4. Insert `[PROMPT]`, `[CONTEXT_PRESENT]`, `[CONTEXT]`, and `[RESPONSE]` only
   after normalization. They are ordinary text, not added tokenizer tokens.
5. Encode Sequence A and Sequence B as an ELECTRA pair with maximum length 512
   and normal `only_first` truncation.
6. When Sequence B exceeds its safe 384-token budget, retain
   `ceil(B/2)` leading and `floor(B/2)` trailing tokens, then allocate the
   remaining pair budget to Sequence A.
7. Resolve the faithful probability column from mutually consistent model
   mappings; never assume column 1.

No code may add tokenizer tokens, resize embeddings, use a remote model ID, or
download at runtime.

## Implemented experiment

- Stage A trains the public 4k split for two epochs and selects one checkpoint
  on the public 1k split by macro F1 at 0.50, validation loss, then earlier
  epoch.
- Arm A performs five fresh official-only BanglaBERT folds.
- Arm B reloads the frozen Stage A checkpoint independently for the same five
  official folds.
- Exact, prompt/context, and high-confidence near-duplicate families share a
  deterministic group. Group-aware stratification is mandatory when any group
  is nontrivial.
- Arm B is promoted only under the locked macro-gain, class-collapse, and
  class-0 guard. Threshold deployment has independent gain, range, and collapse
  guards; otherwise 0.50 remains deployed.
- Every trainable phase receives at most one CUDA OOM fallback, which discards
  partial state and restarts from its original checkpoint with batch size 4 and
  accumulation 4.

The generated notebook embeds a deterministic, hash-verified runtime and needs
exactly the competition input, `abidur14004/new-dataset`, and private model
input `abdullahibnyousuf/banglabert-official-snapshot-9ce791f`. Real metrics,
runtime, model size, and Kaggle score remain pending. This implementation task
did not train, create a local checkpoint/submission, or open competition test
rows.
