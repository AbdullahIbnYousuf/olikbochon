# BanglaBERT Version 3 Results

## Status

**Confirmed successful Kaggle GPU execution and current best scored baseline.**
The self-contained Version 3 notebook completed end to end in **6 minutes 9
seconds** and Kaggle accepted `submission.csv`. The submission received a
public leaderboard score of **0.525**.

This is the strongest recorded project score so far, but Version 3 remains a
baseline experiment rather than a final winning system. One public score does
not establish robustness and must not be used to retune the arm, threshold,
epochs, folds, or preprocessing.

## Kaggle execution record

| Field | Result |
|---|---|
| Notebook | Generated self-contained BanglaBERT Version 3 notebook |
| Selected arm | **Arm A — official-only BanglaBERT** |
| Public 5K adaptation | **Rejected** by the frozen arm-selection rule; Arm B was not deployed |
| Deployed threshold | **0.54** |
| Submitted file | `submission.csv` |
| Fixed-0.50 submission | **Not submitted** |
| Aggregate test prediction distribution | 1,357 label `0`; 1,159 label `1` |
| Total predictions | 2,516 |
| Kaggle runtime | **6 minutes 9 seconds** (369 seconds) |
| Submission result | Completed successfully |
| Public leaderboard score | **0.525** |
| Detailed OOF, Stage A, fold, OOM, and peak-VRAM fields | Not supplied in this result report; do not invent them |

The prediction counts above are aggregate diagnostics only. No IDs, test rows,
probabilities, or individual predictions are stored in Git.

## Scored-baseline comparison

| Version | Model and selected data | Public score | Difference from V1 | Status |
|---|---|---:|---:|---|
| Version 1 | TF-IDF, official labeled sample | 0.466 | — | Previous best; strongest TF-IDF baseline |
| Version 2 | TF-IDF, 5,299 approved labeled rows | 0.426 | -0.040 | Useful negative experiment |
| Version 3 | BanglaBERT Arm A, official-only final arm | **0.525** | **+0.059** | **Current best scored baseline** |

Version 3 improved over Version 1 by **0.059** and over Version 2 by **0.099**.
The public 5K adaptation candidate was evaluated through the precommitted Arm B
path but rejected; the deployed Arm A therefore did not use that adaptation.
This supports the guarded selection design and does not justify public
leaderboard probing or post-hoc threshold changes.

## Submission decision

Only the threshold-0.54 `submission.csv` result was submitted. No fixed-0.50
submission was made. Keep the recorded score as one end-to-end baseline result,
not evidence that Version 3 is the final competition system or a winning model.
