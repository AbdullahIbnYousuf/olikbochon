# অলীকবচন — Local Competition Workspace

Ubuntu workspace for the Kaggle competition **অলীকবচন: Bengali LLM Hallucination Detection Challenge**. Version 3 is the current best scored baseline at **0.525** on the public leaderboard, improving by **0.059** over Version 1's **0.466** TF-IDF result. Its self-contained authenticated BanglaBERT GPU notebook completed in **6 minutes 9 seconds**, selected **Arm A (official-only BanglaBERT)**, rejected the public-5K-adapted Arm B, and deployed threshold **0.54**. Version 2 remains a useful negative TF-IDF experiment at **0.426**. No fixed-0.50 Version 3 submission was made. Version 3 is still a baseline, not the final winning system.

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

Repository tooling may validate file existence, filenames, sizes, hashes, notebook structure, and CSV headers. It must never inspect or display raw test rows or invoke an external inference API. Version 1 used only the official labeled sample. The provided 5,000-row labeled dataset is approved for Versions 2 and 3, but every copy of the competition test file must remain excluded from training, validation, grouping, and model selection.
