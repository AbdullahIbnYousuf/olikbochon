# অলীকবচন — Local Competition Workspace

Setup-only Ubuntu workspace for the Kaggle competition **অলীকবচন: Bengali LLM Hallucination Detection Challenge**. Model development has not started yet.

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

Setup may validate file existence, filenames, sizes, hashes, notebook structure, and CSV headers. It must not inspect or display raw test rows, create predictions, train models, or invoke an external inference API.

