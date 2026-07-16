# অলীকবচন — Local Competition Workspace

Ubuntu workspace for the Kaggle competition **অলীকবচন: Bengali LLM Hallucination Detection Challenge**. Version 1 remains the current best offline TF-IDF baseline at **0.466** on the public leaderboard. Version 2 completed the full Kaggle pipeline successfully using 5,299 unique labeled rows and threshold 0.50, but scored **0.426**, or 0.040 lower than Version 1. This useful negative experiment supports a public-to-competition domain mismatch: more labeled data did not improve leaderboard performance, so Version 2 is not the current final-submission choice and repeated TF-IDF threshold probing is not recommended.

## Directory layout

- `notebooks/original/` — immutable organizer starter notebook and public metadata
- `notebooks/generated/` — notebooks created during later development
- `data/competition/` — private competition files; never committed
- `data/public-20k/` — permitted public training dataset; never committed
- `src/` — reusable source code
- `scripts/` — setup and operational scripts
- `tests/` — automated tests
- `outputs/` — generated results, predictions, and submissions; never committed
- `docs/` — setup records and package versions

## Environment

Activate the environment from the project directory:

```bash
source .venv/bin/activate
```

Verify the setup without displaying raw competition rows:

```bash
python scripts/verify_setup.py
```

Start JupyterLab when notebook work is authorized:

```bash
jupyter lab
```

## Kaggle authentication

Use Kaggle's supported browser authentication flow when credentials need to be created or refreshed:

```bash
kaggle auth login
```

Complete authentication in the browser. Never paste passwords, API tokens, cookies, access tokens, or `kaggle.json` contents into chat or store credentials in this repository.

Private competition access cannot be automated: the participant must open the invitation, join using the intended Kaggle account, and accept the competition rules. No submission is made during setup.

## Scope

Repository tooling may validate file existence, filenames, sizes, hashes, notebook structure, and CSV headers. It must never inspect or display raw test rows or invoke an external inference API. Version 1 used only the official 299-row labeled sample. The provided 5,000-row labeled dataset is approved for Version 2, but every copy of the competition test file must remain excluded from training.
