# Version 4 Preflight Handoff

Date: 2026-07-18

Branch: `research/v4-preflight`
Baseline commit: `ab29d53c8530255a56637526d4edf033c0ebfe3a`

This is a documentation-only preflight. Version 4 was not selected, implemented,
trained, or used for inference. Competition test text was not inspected.

## 1. Repository verification

The repository was cloned from
`https://github.com/AbdullahIbnYousuf/olikbochon.git` because a read-only SSH
authentication check returned `Permission denied (publickey)`. After `git fetch
origin`, local `main`, `origin/main`, and `HEAD` all resolved to the expected commit
`ab29d53c8530255a56637526d4edf033c0ebfe3a`. The tree was clean. Work then moved to
the new branch `research/v4-preflight`; `main` was not modified.

Initial verification results:

| Command | Result |
|---|---|
| `ruff check .` | Not run: `ruff` is not recognized as a command |
| `pytest -q` | Not run: `pytest` is not recognized as a command |
| `python scripts/verify_setup.py` | Failed at import: `ModuleNotFoundError: No module named 'nbformat'` |
| `git status` | Clean on `research/v4-preflight` before this report |
| `git diff --check` | Passed |

The repository's `pyproject.toml` declares Ruff target `py314`; `SETUP_STATUS.md`
records the original environment as Python 3.14.4. The documented local method is to
activate an already-created `.venv` (`source .venv/bin/activate`), with setup package
versions recorded in `docs/setup-package-versions.txt` (`ruff==0.15.21`,
`pytest==9.1.1`, and `nbformat==5.10.4`). This Windows clone has no such environment,
and the active Python is 3.12.6. There is no repository-wide lock file or documented
fresh-environment install command. No package was installed and no environment was
created during preflight.

## 2. Architecture and Version 1–3 summary

The tracked repository is organized as follows:

- `src/olikbochon/`: data discovery/loading, preprocessing, metrics, duplicate
  controls, submission validation, and version-specific V2/V3 pipelines.
- `src/ftfy/`, `src/wcwidth/`, and the Bangla normalizer: vendored offline V3 runtime.
- `notebooks/original/`: immutable organizer notebook metadata.
- `notebooks/generated/`: generated V1 TF-IDF, V2 TF-IDF, and V3 BanglaBERT notebooks.
- `scripts/`: setup verification and deterministic V3 notebook/vendor generation.
- `tests/`: 20 safety, data, metric, pipeline, and V3 runtime test modules.
- `docs/`: plans, audits, runbooks, results, licenses, validation, and risks.

Verified project records:

| Version | System and data | Public score | Finding |
|---|---|---:|---|
| V1 | Word/character TF-IDF plus logistic regression; official sample | 0.466 | Strongest sparse baseline |
| V2 | Same family adapted with approved public 5K | 0.426 | Negative transfer/domain mismatch |
| V3 | BanglaBERT; official-only selected arm | **0.525** | Current best; public-5K arm rejected; threshold 0.54 |

V3 completed on Kaggle in 6 minutes 9 seconds. Its offline bundle, normalizer checks,
grouped validation, selection guards, and submission safety are reusable foundations,
but the small official training set and optimistic OOF model/threshold selection remain
important limitations.

## 3. Hardware report

Only non-secret machine facts are recorded here.

| Item | Verified value |
|---|---|
| OS | Windows Pro, display version 25H2, build `10.0.26200.8875`, 64-bit |
| CPU | Intel Core i7-14700K; 20 cores / 28 logical processors (provided and previously verified) |
| RAM | 31.72 GB (provided and previously verified) |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| VRAM | 12,282 MiB total; approximately 10,420 MiB free during audit |
| Driver / compatibility | NVIDIA 591.86; driver-reported CUDA 13.1 |
| GPU state | WDDM display mode; about 16–24% utilization and 1.5 GiB in use during checks |
| Workspace disk | `F:` 634.76 GiB total, 193.52 GiB free |
| Python | 3.12.6 at `C:\Python312\python.exe` |
| Python 3.11 | Not available through the Python launcher |
| GitHub CLI | Not installed |

`nvidia-smi` showed only desktop/GUI processes and several processes hidden by Windows
permissions; no competing compute training job was identified. PyTorch is not installed.
The driver-reported CUDA compatibility does not imply that a CUDA Toolkit or a compatible
PyTorch build is installed.

GitHub access findings:

- Public HTTPS clone/fetch works.
- SSH authentication is unavailable (`Permission denied (publickey)`).
- A non-mutating `git push --dry-run origin research/v4-preflight` reached GitHub but
  returned HTTP 403: the current HTTPS credential has no write permission to this repo.
- Therefore this machine cannot presently push the branch or open a PR through the CLI.

## 4. Exact metric evidence

### Verified primary evidence

The repository contains the organizer-supplied rulebook. Its scoring clause says:

> **Primary metric**: Macro F1 on the HALLUCINATED class (label = 0).

It separately says that higher F1 on the C1 cultural-distance subset is a **tie-breaker**,
not part of the primary score. Therefore there is no evidence that the leaderboard score
is culturally weighted or subset-weighted.

The organizer starter notebook explicitly computes
`f1_score(y_true, y_pred, average='macro')`, labels that value the primary metric, and
separately prints the label-0 and label-1 F1 values. That implementation is ordinary
unweighted two-class macro F1.

Official page locations (access requires the competition session):

- Evaluation: <https://www.kaggle.com/competitions/bengali-hallucination/overview/evaluation>
- Rules: <https://www.kaggle.com/competitions/bengali-hallucination/rules>
- Discussions: <https://www.kaggle.com/competitions/bengali-hallucination/discussion>

### Conclusion and remaining ambiguity

The best implementation evidence available in this audit is **ordinary two-class macro
F1**, because the organizer starter code implements it exactly. However, the rulebook's
phrase “on the ... class (label = 0)” is internally inconsistent with the standard meaning
of macro F1 and could instead intend binary label-0 F1. No evaluator source code or
organizer clarification was accessible, so the exact hidden scorer is not conclusively
verified. Continue reporting both class F1 values and macro F1, retain macro F1 as the
provisional objective, and do not tune from leaderboard feedback.

Required organizer clarification: provide either the scorer code or an explicit equivalent,
for example `f1_score(y_true, y_pred, average="macro")` versus
`f1_score(y_true, y_pred, average="binary", pos_label=0)`.

## 5. Public-notebook comparison

No notebook can be responsibly listed as verified in this audit. General web search did
not index the invitation-only competition's code pages. The in-app signed-session browser
failed before navigation with a local sandbox permission error (`EPERM ... lstat ...
AppData`). Kaggle CLI/authentication was not available. Consequently, scores near 0.68,
authors, notebook URLs, code, data, validation, and runtime claims could not be checked.

This is an access blocker, not evidence that such notebooks do not exist. To complete the
audit manually, provide public notebook URLs or exported notebook files plus a leaderboard
screenshot/record showing the score. Do not provide credentials, cookies, tokens, test
data, or private notebook content. Each supplied notebook should then be classified into:

1. verified page/code facts;
2. author-reported score or method claims; and
3. technical inference about leakage, overfitting, runtime, and reproducibility.

Until that evidence exists, no notebook-derived method should be copied or treated as a
0.68-validated result.

## 6. Recommended V4 options

These are planning estimates, not measured V4 results. Times assume mixed precision,
sequence length roughly 256–384, the current small-data scale, and a healthy native-Windows
PyTorch environment on the RTX 4070 SUPER. Cross-validation multiplies training time.

| Rank | Option | Expected value | Complexity | Validation risk | Min / recommended VRAM | Approx. train / inference | Checkpoint | Windows / WSL2 / Kaggle offline |
|---:|---|---|---|---|---|---|---|---|
| 1 | Strengthen BanglaBERT with context-aware serialization, conservative truncation, and grouped repeated validation | High: closest extension of the 0.525 winner | Medium | Medium-high due to 299-row official set | 6 / 10 GB | 10–30 min per fit; 1–4 min inference | 0.45–1.5 GB | Native good; WSL2 not material; offline Kaggle yes with existing snapshot |
| 2 | BanglaBERT + sparse V1 blend using OOF-only calibrated weights | Medium-high if errors are complementary | Medium-high | High: blend weight can overfit tiny OOF | 6 / 10 GB | 15–45 min total; 2–6 min inference | 0.5–1.7 GB | Native good; WSL2 not material; offline Kaggle yes |
| 3 | XLM-R base challenger with identical groups and frozen comparison rule | Medium; useful architecture diversity | Medium | High: heavier model and small validation | 9 / 12 GB | 25–70 min per fit; 3–8 min inference | 1.1–2.5 GB | Native feasible with small batches; WSL2 mildly helpful for tooling only; offline Kaggle yes if pinned |
| 4 | MuRIL base challenger for Bengali/transliteration | Medium-low pending evidence of code-mixing benefit | Medium-high | High | 9 / 12 GB | 25–75 min per fit; 3–8 min inference | 0.95–2.3 GB | Native feasible; WSL2 mildly helpful only; offline Kaggle yes if pinned |
| 5 | Parameter-efficient adaptation or compact two-model ensemble | Uncertain; possible diversity gain | High | Very high without larger trustworthy validation | 10 / 12 GB | 45–150 min; 5–15 min inference | 1.5–4 GB | Native possible but more fragile; WSL2 may ease some libraries; offline Kaggle possible with full vendoring |

Typical peak system RAM is estimated at 8–16 GB for BanglaBERT and 12–24 GB for XLM-R,
MuRIL, or two-model work. Working storage should reserve 10–20 GB for one model's cached
snapshot, optimizer/checkpoint, notebook artifacts, and safety margin; 25–40 GB is prudent
for a two-model comparison. Final inference artifacts should exclude optimizer states and
remain far below the 50 GB competition limit.

Rank 1 has the best expected-value-to-risk ratio because it preserves the verified offline
V3 stack and changes only scientifically motivated representation/validation choices.
Rank 2 is attractive only if frozen OOF evidence shows complementary errors without class
collapse. Ranks 3–5 should be gated behind a clear validation plan and metric clarification.

## 7. Compatibility and reproducibility

Native Windows should support all ranked options with an appropriate CUDA-enabled PyTorch
wheel and pinned packages. WSL2 would mainly improve Linux parity and shell tooling; it is
not required for the models and would not materially improve expected model quality. Do not
install it solely for V4.

Every option can be reproduced with Kaggle internet disabled only if the exact base model,
tokenizer, normalizer, configuration, licenses/notices, and inference code are attached and
loaded locally. The existing V3 bundling and hash-validation pattern should remain the
template. External APIs are forbidden at inference.

## 8. Validation-overfitting and leakage risks

- The official labeled sample is tiny; single-split gains and tuned thresholds are unstable.
- Reusing one OOF table for model, threshold, blend, and rule selection compounds optimism.
- The public 5K already produced negative transfer and has unresolved license/provenance;
  do not assume more of it improves V4.
- Exact and near-duplicate prompt/context families must remain grouped across folds.
- Public aggregate/subset duplication and the known quarantined test copy are critical
  leakage hazards.
- Context-present and context-absent regimes may reward different shortcuts; report both.
- Cultural-subset performance is a tie-break diagnostic, not a license to tune hidden
  subset weights.
- Public notebook scores and leaderboard scores are not validation evidence unless their
  data flow, splits, and submission history can be audited.
- Any ensemble weights, thresholds, rules, or checkpoint choices must be selected inside
  training-only folds and frozen before final inference.

## 9. Concrete blockers and missing access

1. Exact evaluator code or an explicit organizer clarification is missing.
2. The official Kaggle pages/discussions and public notebooks were inaccessible in this
   environment; manual URLs or exports are required for the notebook audit.
3. Local verification dependencies and the declared Python 3.14 environment are absent.
4. The current GitHub credentials receive HTTP 403 for repository push; SSH has no accepted
   key, and `gh` is unavailable.
5. The public 5K dataset's provenance/license and intended version remain unresolved.

## 10. Decisions requiring approval

- Approve creation of a repository-compatible Python 3.14 environment and an exact install
  command before any package installation. A lock/requirements source should be agreed
  first; the historical package-version record alone may be broader than necessary.
- Provide or authorize access to official Kaggle metric clarification and public notebook
  URLs/exports.
- Restore GitHub write permission for this repository or push the branch manually.
- Choose one V4 option only after the metric and notebook evidence gates are addressed.
- Decide whether the public 5K may be used further after provenance/license confirmation.

No V4 implementation, model download, training, prediction, test inspection, or submission
was performed.
