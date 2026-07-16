# Run TF-IDF Baseline Version 1 on Kaggle

## Before uploading

Use the generated notebook:

`notebooks/generated/tfidf_baseline_v1.ipynb`

The canonical reviewable source is `notebooks/generated/tfidf_baseline_v1.py`. Do not upload or execute the original starter notebook for this run.

## Kaggle steps

1. Open Kaggle and choose **Create → New Notebook**, then use **File → Import Notebook** to import `tfidf_baseline_v1.ipynb`.
2. Attach only the official `bengali-hallucination` competition data through **Add Input**. Do not attach the unresolved public dataset.
3. Open **Notebook options** and set **Internet off**.
4. Set the accelerator to **None/CPU**. A GPU is unnecessary.
5. Confirm the input panel contains the organizer files. The notebook recursively discovers `dataset samples.json`, `test set.csv`, and either `sample submission.csv` or `sample_submission.csv`.
6. Choose **Run All**. Do not add package-install, download, API, or display cells.
7. Review only the safe aggregate logs: selected paths, schemas, row/count metadata, validation metrics, threshold table, prediction-label counts, and submission shape. Do not display test or submission rows.
8. Confirm the final log says the output location is `/kaggle/working/submission.csv` and the columns are exactly `id,label`.
9. Choose **Save Version** and wait for the committed run to finish successfully with internet disabled.
10. In the saved version’s **Output** panel, locate `submission.csv` and use **Submit to Competition**.

## Required success checks

Before submitting, confirm from aggregate logs that:

- the official labeled file hash validation passed;
- both labeled classes were present;
- the discovered sample-submission template matched the test ID sequence when present;
- test IDs had no missing values;
- the final submission shape matched the test row count dynamically;
- output columns were exactly `id,label`;
- labels were integers restricted to `0` and `1`;
- no exception, network request, or missing-file fallback occurred; and
- total runtime remained comfortably under nine hours.

The notebook deliberately does not print IDs, individual predictions, test prompts, contexts, responses, or submission rows.

## Record the run

Record these items in the team experiment log without committing data or predictions:

- Git commit and notebook version;
- Kaggle saved-version number;
- runtime and CPU environment;
- selected deployment threshold;
- validation summary;
- public leaderboard score; and
- any warning/error text that contains no test content.

Treat the first leaderboard score as a coarse pipeline sanity check. Do not change the model or threshold based on one public score. The provisional threshold strongly favors label 0 because the official metric wording remains unresolved; obtain organizer clarification before interpreting it as the correct final decision rule.
