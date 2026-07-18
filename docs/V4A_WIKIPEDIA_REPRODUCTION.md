# Version 4-A Wikipedia Retrieval Reproduction

## Status and purpose

Version 4-A is a clean, independent reproduction of the lexical/Wikipedia idea
from public Kaggle notebook `nazifaanjum/notebook42f62bbcad`, Version 4. The
public source result is verified at **0.685**. V4-A itself has not been run or
submitted on Kaggle, so no project score is claimed.

The implementation uses only the official competition attachment and the
pinned Wikipedia dataset. It does not use the source notebook's inaccessible
private competition-data copy, fallback IDs, file enumeration, or saved code.

## Pinned Wikipedia input

| Field | Pin |
|---|---|
| Kaggle dataset | [`abyaadrafid/bnwiki`](https://www.kaggle.com/datasets/abyaadrafid/bnwiki) |
| Dataset title | Bangla Wikipedia Articles |
| Kaggle dataset ID | `228152` |
| Version | `1` |
| Kaggle metadata update | 2019-06-11 18:06:12 UTC |
| Kaggle metadata license | `CC0-1.0` |
| Mounted chunk files | 301 |
| Mounted bytes | 312,927,965 |
| Content-manifest SHA-256 | `052ce8d9061de8d1f3c9a4cd6814f9c54b6cc92953546845bc0595767c23bcc2` |

The actual Kaggle dataset exposes one logical root, `lolol`, containing 301
WikiExtractor chunks beneath `AA`, `AB`, `AC`, and `AD`. V4-A authenticates
those exact paths, their total size, and every file's content digest. Articles
are deduplicated by URL. No Wikipedia passage is printed or written to project
outputs.

The downloadable ZIP currently contains a second byte-identical historical
tree at `lolol/lolol`, which explains the earlier doubled local count of 602.
That duplicate archive tree is not part of the logical Kaggle-mounted manifest,
is rejected as an extra directory, and is never required or indexed by V4-A.

The Kaggle publisher describes the material as a processed Bengali Wikipedia
dump and labels the dataset CC0. Underlying Wikimedia text remains subject to
the applicable [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use).
Keep attribution to Bengali Wikipedia and the Kaggle dataset in any report or
derived public artifact.

## Frozen method

- Clean null context using the original method's null-value semantics.
- Convert Bengali digits to ASCII for numeric comparisons.
- Build the same eight lexical features in a fixed order.
- Index Wikipedia title plus the first 800 body characters using character
  `char_wb` TF-IDF n-grams 2–4 with at most 50,000 features.
- Retrieve the single highest-similarity article for each prompt.
- Preserve real competition context. For missing context, accept the retrieved
  snippet only when similarity is at least 0.25.
- Standardize features and fit balanced logistic regression with seed 42 and
  `max_iter=1000`.
- Locate the label-1 probability column from `model.classes_`.
- Predict label 1 at probability 0.50; label 0 remains hallucinated.

## Two validation estimates

The Kaggle run must report both estimates separately:

1. **Original-method full-OOF retrieval-threshold tuning estimate.** Five-fold
   stratified OOF on only the official no-context subset, trying retrieval
   cutoffs 0.05 through 0.40 on the same OOF result. This reproduces the public
   approach and is explicitly optimistic.
2. **Nested 5x3 duplicate-aware grouped estimate.** Exact normalized rows and
   every response sharing normalized prompt/context form one family. Each outer
   validation family is untouched while its retrieval cutoff is selected by
   three-fold grouped CV inside the corresponding outer-training portion.

The published 0.25 retrieval cutoff remains frozen for final V4-A artifacts;
the public leaderboard is not used for local threshold selection.

## Runtime outputs

A successful internet-off Kaggle run creates:

- `/kaggle/working/submission.csv` — schema- and ID-validated candidate, not yet submitted;
- `/kaggle/working/v4a_oof_probabilities.csv` — honest grouped OOF probabilities;
- `/kaggle/working/v4a_test_probabilities.csv` — ID-aligned label-1 probabilities for a future V3 ensemble; and
- `/kaggle/working/v4a_run_summary.json` — aggregate configuration and validation results.

These are ignored runtime outputs. They must not be committed, printed, or
uploaded outside the permitted Kaggle competition workflow.
The aggregate run summary records the Kaggle Python, NumPy, pandas, and
scikit-learn versions because small tokenizer, TF-IDF, or solver changes can
affect exact reproduction.

## Kaggle preflight

1. Import `notebooks/generated/wikipedia_retrieval_v4a.ipynb`.
2. Attach the official Bengali hallucination competition input.
3. Attach `abyaadrafid/bnwiki`, version 1.
4. Select CPU and keep internet disabled.
5. Run from a fresh session and verify the input-manifest authentication.
6. Record both validation estimates and runtime without submitting.

Version 4-A does not modify V3 and does not yet define a V3/V4 ensemble.
