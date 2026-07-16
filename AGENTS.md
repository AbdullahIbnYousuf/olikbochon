# Permanent Project Instructions

This repository is for the IUT ICT Fest 2026 Bengali LLM Hallucination Detection Challenge.

- Label `0` means hallucinated.
- Label `1` means faithful.
- A final submission must contain exactly the columns `id,label`.
- Final Kaggle inference must not call an external API.
- Never manually label, inspect, print, summarize, or display competition test examples.
- Metadata, filenames, sizes, hashes, and CSV headers may be inspected.
- Do not hardcode IDs, row counts, row order, labels, or test-specific values.
- Keep everything in `notebooks/original/` immutable.
- Put generated notebooks in `notebooks/generated/`.
- Put reusable code in `src/` and tests in `tests/`.
- Keep all data, outputs, submissions, and model weights ignored by Git.
- Final inference must be reproducible with Kaggle internet disabled.
- Final inference runtime must remain under nine hours on a P100 or two T4 GPUs.
- Combined model weights must remain under 50 GB.
- Use only publicly permitted data and declare and cite all external data.
- Never expose secrets, tokens, cookies, `kaggle.json`, or Kaggle credentials.
- Do not upload competition data anywhere.

