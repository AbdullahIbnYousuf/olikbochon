# Run TF-IDF Baseline Version 2 on Kaggle

## Recorded outcome

The real Version 2 Kaggle run completed successfully. It trained the final model on 5,299 unique labeled rows and submitted `submission.csv` at threshold 0.50. The public leaderboard score was **0.426**, which is **0.040 below** Version 1's **0.466**.

The end-to-end pipeline worked, including file-role discovery, safe audits, offline model fitting, test inference, submission validation, and submission. The lower score is a useful negative experiment, not an implementation failure. It supports the previously observed domain mismatch and shows that adding the public labeled data did not improve leaderboard performance.

Version 1 remains the current best scored submission. Version 2 should not currently be selected as the final scored submission, and repeated TF-IDF threshold probing is not recommended. A future Version 3 should investigate a stronger context-aware or multilingual transformer model; no Version 3 work is part of this run record.

## Inputs

1. Import `notebooks/generated/tfidf_baseline_v2.ipynb` into a new Kaggle notebook.
2. Attach the official Bengali hallucination competition input.
3. Attach the Kaggle dataset `abidur14004/new-dataset`.
4. Do not attach extra copies of either dataset.

## Notebook settings

1. Set **Internet** to **Off**.
2. Set **Accelerator** to **None/CPU**.
3. Do not add install, download, API, display, or exploratory test-data cells.

## Run and save

1. Choose **Run All**.
2. Confirm safe logs identify the public 4k and 1k roles and a separate coherent competition root.
3. Confirm exact composition, conflict, overlap, and near-duplicate audits pass before modeling.
4. Confirm the public bundled `test set.csv` is below a quarantined public root and is not selected.
5. Confirm the public-validation result is labeled `Public-validation threshold-tuning estimate` and the official result is labeled `Independent official-sample evaluation`.
6. Confirm the selected threshold is frozen before final refit.
7. Save a committed Kaggle version and wait for the fresh run to complete with internet disabled.

## Outputs and submission

The notebook independently validates and writes:

- `/kaggle/working/submission.csv` using the public-validation macro-F1-selected threshold;
- `/kaggle/working/submission_fixed_050.csv` using threshold 0.50.

No class-0 experimental output is created. Check only aggregate shape, exact columns, label domain, ID-order validation status, and label counts. Never display submission rows, IDs, individual predictions, or test text.

For reproducing the recorded Version 2 experiment, `submission.csv` is the selected-threshold output. Record the saved notebook version, Git commit, CPU runtime, selected threshold, and Kaggle score without committing the submission file. Do not use repeated submissions to probe alternative TF-IDF thresholds.

## Accepted limitation

The final model includes the former public validation and official evaluation rows, but the threshold stays frozen at the value selected before that refit. This can shift calibration slightly. Do not retune from the official sample, competition test, or leaderboard.
