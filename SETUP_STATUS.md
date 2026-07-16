# Setup Status

This file is updated as setup progresses. It intentionally contains no raw dataset content.

## Current state

- Host prerequisites inspected and the project structure created without deleting existing files.
- `.venv` repaired in place with Python 3.14.4 and pip 26.1.2.
- Setup-only packages installed; exact versions recorded in `docs/setup-package-versions.txt`.
- Kaggle OAuth authentication and access to `bengali-hallucination` verified.
- Starter notebook and metadata downloaded into `notebooks/original/`; the notebook was validated without execution, hashed, and made read-only.
- Competition and permitted public dataset archives downloaded, path-audited, integrity-tested, and safely extracted without overwriting; ZIP archives retained.
- `scripts/verify_setup.py` passed with all mandatory artifacts and the optional sample submission available.
- The workspace is initialized as a Git repository with data, outputs, credentials, ZIPs, submissions, and weights excluded from version control.
- Version 1 used only the 299-row official labeled sample and completed a fresh Kaggle **Save & Run All** on CPU with internet disabled and no external API or model download.
- Kaggle runtime was 53 seconds. The validated `submission.csv` used `macro_f1_oof` threshold 0.53 and received a public leaderboard score of 0.466; the fixed-0.50 and class-0 experimental files were not submitted.
- The successful submission confirms the complete Version 1 pipeline works. One public score is only a baseline and does not justify model-quality conclusions or leaderboard threshold probing.
- The provided 5,000-row labeled dataset is approved for Version 2. Version 2 has not started, and every copy of the competition test file remains excluded from training.
