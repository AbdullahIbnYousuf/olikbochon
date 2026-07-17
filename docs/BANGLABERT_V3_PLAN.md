# BanglaBERT Version 3 Plan

## Current status

The technical foundation is ready, but Version 3 training has not started.

- The official BanglaBERT snapshot is pinned, authenticated, private on Kaggle,
  and verified for local-only loading.
- The tokenizer, pretraining architecture, and downstream classification
  transition pass CPU-only synthetic smoke tests.
- The official normalizer is pinned, minimally vendored, and parity-tested.
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

## Next authorized phase

The next task may implement the previously approved two same-backbone training
arms and offline Kaggle notebook. It must separately authorize training and must
retain the existing data quarantine, duplicate controls, validation rules,
runtime limits, offline flags, and private-weight handling. This task did not
train, create checkpoints, run competition-test inference, or create a
submission.
