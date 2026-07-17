# Run BanglaBERT Version 3 on Kaggle

## Confirmed Version 3 run

The generated Version 3 notebook completed successfully on Kaggle in **6
minutes 9 seconds**. The frozen selection logic chose **Arm A, official-only
BanglaBERT**, rejecting the public-5K-adapted Arm B. The deployed threshold was
**0.54**.

Kaggle accepted `submission.csv` and returned a public leaderboard score of
**0.525**, making Version 3 the current best scored project baseline. This is
**0.059 above** Version 1's 0.466. The aggregate test prediction distribution
was 1,357 label `0` and 1,159 label `1`; no row-level predictions or IDs are
recorded. No fixed-0.50 submission was made.

This confirms the complete offline transformer pipeline works on Kaggle. It
does not establish that Version 3 is a final winning system, and the public
score must not be used to revise the frozen arm, threshold, training, or
preprocessing decisions.

## Create the notebook

1. Import `notebooks/generated/banglabert_v3.ipynb` into Kaggle.
2. Attach the official competition input.
3. Attach `abidur14004/new-dataset`.
4. Attach the private dataset
   `abdullahibnyousuf/banglabert-official-snapshot-9ce791f`, version 1.
5. Confirm those three required resources are attached. Kaggle may mount them
   beneath nested competition/dataset directories; the notebook resolves each
   role recursively and rejects missing or ambiguous roots.
6. Select a T4 GPU and keep Internet **Off**.

Do not add install, download, API, exploratory display, or alternate-model
cells. The notebook intentionally fails without CUDA or when any input,
authenticated digest, runtime bundle, or normalizer startup check differs.

## Run all cells

Run all cells from a fresh session. Safe progress includes environment/model
authentication, normalizer digest validation, aggregate data/group audits,
Stage A epochs, five Arm A folds, five Arm B folds, frozen arm/threshold rules,
final training, and aggregate test inference. It must never display test text,
IDs, probabilities, or individual predictions.

If a training phase raises CUDA OOM, the notebook discards every partially
updated object and automatically restarts that complete phase once with batch
size 4 and accumulation 4. If the restarted phase also fails, stop; do not edit
other hyperparameters or resume partial weights.

## Confirm outputs

Successful execution creates:

- `/kaggle/working/submission.csv` using the frozen deployed threshold;
- `/kaggle/working/submission_fixed_050.csv` only when deployment is not 0.50;
- `/kaggle/working/banglabert_v3_model/` containing the private final model,
  tokenizer, metadata, and attribution; and
- `/kaggle/working/v3_run_summary.json` containing aggregate/configuration
  information only.

Confirm the printed schema, row-count, ID-sequence, and label-domain validation
messages. Save a committed notebook version, wait for that fresh internet-off
run to finish, and keep the model output private.

The recorded run submitted only `submission.csv` at threshold 0.54; the
fixed-0.50 reference was not submitted. For any reproducibility rerun, record
the runtime, selected arm, threshold, model size, OOM status, and public score
without committing outputs. Do not submit the fixed reference as a leaderboard
probe and do not change any decision using the public score.
