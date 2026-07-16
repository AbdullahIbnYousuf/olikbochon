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
8. Confirm the final log reports all three independently validated output paths and exact `id,label` columns:
   - `/kaggle/working/submission.csv` — `macro_f1_oof`, threshold 0.53 in the current run, and the provisional default;
   - `/kaggle/working/submission_fixed_050.csv` — `fixed_050`, threshold 0.50, and the clean reference;
   - `/kaggle/working/submission_class0_experimental.csv` — `class0_f1_oof_experimental`, currently threshold 0.71.
9. Choose **Save Version** and wait for the committed run to finish successfully with internet disabled.
10. In the saved version’s **Output** panel, use either `submission.csv` or `submission_fixed_050.csv` as the reasonable Version 1 candidate.

Do not submit `submission_class0_experimental.csv` unless the organizers explicitly confirm that the evaluator is binary F1 with `pos_label=0`. It predicts almost every validation row as label 0 and is retained only as an experimental metric diagnostic.

## Required success checks

Before submitting, confirm from aggregate logs that:

- the official labeled file hash validation passed;
- both labeled classes were present;
- the discovered sample-submission template matched the test ID sequence when present;
- test IDs had no missing values;
- every generated submission shape matched the test row count dynamically;
- every output had exactly `id,label` columns;
- every output contained only integer labels restricted to `0` and `1`;
- no exception, network request, or missing-file fallback occurred; and
- total runtime remained comfortably under nine hours.

The notebook deliberately does not print IDs, individual predictions, test prompts, contexts, responses, or submission rows.

## Record the run

Record these items in the team experiment log without committing data or predictions:

- Git commit and notebook version;
- Kaggle saved-version number;
- runtime and CPU environment;
- all three named threshold strategies and selected thresholds;
- validation summary;
- public leaderboard score; and
- any warning/error text that contains no test content.

Treat a leaderboard score only as a coarse pipeline sanity check. Public-leaderboard threshold probing is prohibited: do not compare these files to select a threshold, and do not alter the model or decision rule based on leaderboard movement. Macro-F1 thresholding is the provisional default because it keeps both classes meaningful; fixed 0.50 is the clean reference. Organizer clarification is still required before treating either objective as the official one.
