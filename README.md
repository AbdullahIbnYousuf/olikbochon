# অলীকবচন — Local Competition Workspace

Ubuntu workspace for the Kaggle competition **অলীকবচন: Bengali LLM Hallucination Detection Challenge**. Version 4-A is the current best **project** baseline at **0.685** on the public leaderboard. Its offline Wikipedia-retrieval notebook completed in **1 hour 25 minutes 38 seconds** and produced an honest duplicate-aware grouped macro F1 estimate of **0.6872701508**. Version 3 remains the strongest transformer baseline at **0.525**; Version 1 scored **0.466**, and Version 2 remains a useful negative TF-IDF experiment at **0.426**.

Public Kaggle screenshots verify substantially stronger external results:
`olikbochon_fullsqueeze` Version 6 scored **0.684**, and
`notebook42f62bbcad` Version 4 scored **0.685**. Their scores are confirmed,
while their validation quality and exact reproducibility are separate issues.
The clean V4-A branch independently reproduces the latter's Wikipedia/lexical
method with official inputs, dynamically validated Wikipedia data, grouped
validation, and preserved Git-ignored probability artifacts. Project V4-A
Version 2 independently achieved the same **0.685** public score.

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
