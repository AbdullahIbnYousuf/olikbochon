# Version 4-A Wikipedia Retrieval Reproduction

## Status and purpose

Version 4-A is a clean, independent reproduction of the lexical/Wikipedia idea
from public Kaggle notebook `nazifaanjum/notebook42f62bbcad`, Version 4. The
public source result is verified at **0.685**. Project notebook
`abdullahibnyousuf/v4-a-dynamic-wikipedia-discovery`, Version 2, completed and
independently scored **0.685**. V4-A is therefore the current best project
result, while remaining a baseline rather than a final winning system.

The implementation uses only the official competition attachment and the
declared Wikipedia dataset. It does not use the source notebook's inaccessible
private competition-data copy, fallback IDs, file enumeration, or saved code.

## Confirmed Kaggle result

| Result | Value |
|---|---:|
| Successful notebook version | 2 |
| Public leaderboard score | **0.685** |
| Kaggle displayed runtime | **1h 25m 38s** |
| Internal pipeline timer | 5,113.533819 seconds |
| Honest grouped validation macro F1 | **0.6872701508** |
| Honest grouped validation label-0 F1 | **0.7102803738** |
| Test prediction distribution | label 0: 1,664; label 1: 852 |
| Embedded runtime archive SHA-256 | `9dd66e6f70223adbaa3bd84a52f0c67d5009ea3aed8e627874cb1bd4886fc63c` |

The displayed runtime includes Kaggle orchestration overhead; the internal
timer starts inside the pipeline, so the two timing values are expected to
differ slightly. The public score is one leaderboard observation and does not
remove the validation limitations documented below.

## Dynamically validated Wikipedia input

| Field | Declared source |
|---|---|
| Kaggle dataset | [`abyaadrafid/bnwiki`](https://www.kaggle.com/datasets/abyaadrafid/bnwiki) |
| Dataset title | Bangla Wikipedia Articles |
| Kaggle dataset ID | `228152` |
| Version | `1` |
| Kaggle metadata update | 2019-06-11 18:06:12 UTC |
| Kaggle metadata license | `CC0-1.0` |

For competition-speed and mount compatibility, V4-A does not require an exact
per-file cryptographic manifest. It recursively finds coherent roots containing
`AA`, `AB`, `AC`, and `AD`, accepts only direct `wiki_<number>` chunks, and
requires at least 250 chunks. A single layout is used directly. If Kaggle exposes
both the single and nested archive layouts, their relative paths, sizes, and
SHA-256 digests must be byte-identical before one copy is discarded; distinct
corpora remain an error. Every selected chunk must be valid UTF-8 containing
JSON-object records before retrieval begins. Articles are then deduplicated by
URL. No Wikipedia passage is printed or written to project outputs.

The discovered canonical file count, aggregate bytes, and number of discarded
duplicate paths are recorded in the safe run summary. This makes corpus drift
visible without making an exact historical mount shape a Kaggle startup blocker.

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

- `/kaggle/working/submission.csv` — schema- and ID-validated scored submission;
- `/kaggle/working/v4a_oof_probabilities.csv` — honest grouped OOF probabilities;
- `/kaggle/working/v4a_test_probabilities.csv` — ID-aligned label-1 probabilities for a future V3 ensemble; and
- `/kaggle/working/v4a_run_summary.json` — aggregate configuration and validation results.

These are ignored runtime outputs. They must not be committed, printed, or
uploaded outside the permitted Kaggle competition workflow.
The aggregate run summary records the discovered Wikipedia count and bytes plus
the Kaggle Python, NumPy, pandas, and scikit-learn versions because corpus or
runtime changes can affect reproduction.

The successful Version 2 outputs are preserved locally under the Git-ignored
`artifacts/kaggle/v4a_0685/` directory:

| Artifact | SHA-256 |
|---|---|
| `submission.csv` | `bdfcd907177fb35537ae33998c6427f65d3ce574d18f82ee3e656bdb4ee2fa19` |
| `v4a_oof_probabilities.csv` | `78ee80a3f9944500090342170ccf60db7cbc793c7fbe6a17ad3e6a6906d51ea6` |
| `v4a_test_probabilities.csv` | `2dd5ffa97c285727143fefe7d172bd02588368b7c3ff111088709ce0eb27adc2` |
| `v4a_run_summary.json` | `b30dabf16cacf0bb684791740cf1d486b73bade27ff6805edb9daf0bccffca85` |

See `docs/EXPERIMENT_ARTIFACT_INVENTORY.md` for the complete V1–V4 audit.

## Kaggle preflight

1. Import `notebooks/generated/wikipedia_retrieval_v4a.ipynb`.
2. Attach the official Bengali hallucination competition input.
3. Attach `abyaadrafid/bnwiki`, version 1.
4. Select CPU and keep internet disabled.
5. Run from a fresh session and verify the dynamic Wikipedia discovery and parse summary.
6. Record both validation estimates and runtime; do not create another
   leaderboard submission without a separately justified experiment.

Version 4-A does not modify V3 and does not yet define a V3/V4 ensemble.
