# V14 hallucinated-class F1 correction

## Authenticated metric

The local `ICT Fest Datathon Rulebook 2026.md` identifies label `0` as hallucinated and
states that the primary metric is F1 on the hallucinated class. The operational scorer is
therefore `f1_score(y_true, y_pred, pos_label=0, zero_division=0)`. Macro F1, label-1 F1,
and accuracy are diagnostics only. No authenticated local competition document contradicted
this mapping.

V13 was preserved first in commit `4566dda` (`Record rejected V13 public template transfer`)
and pushed normally to `research/v4-banglabert-context`. It records that every public-transfer
gate failed, zero public-to-test transfers were eligible, no submission was generated, and no
further V13 experiment was run. PR #4 was not merged.

## Frozen inputs and validation

- official labeled sample: 299 rows, corrected routes 130 present / 169 absent;
- validation: seeds 17, 29, and 43; five strengthened grouped outer folds per seed;
- threshold selection: three grouped inner folds within every outer-training partition;
- threshold grid: 0.10 through 0.90 in increments of 0.05;
- every probability is `P(label 1)`; values below threshold predict hallucination label `0`;
- outer row overlap: zero; outer strengthened-group overlap: zero;
- inner OOF coverage: complete; outer-validation rows used for selection: zero;
- mDeBERTa revision: `b5113eb38ab63efdd7f280f8c144ea8b13f978ce`;
- mDeBERTa weight SHA-256:
  `7c8e29f1115986d032e92b0fbaa0bdef1062a46f658b08705f237c05014a8541`;
- Bengali Wikipedia logical manifest:
  `af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf`.

No public labeled data, leaderboard score, Qwen, XLM-R, test label, or raw test text was used
for fitting or selection.

## Genuine repeated OOF audit at threshold 0.50

All rows below are pooled across three genuine OOF repetitions (897 predictions). Confusion
matrices use true rows and predicted columns in label order `[0, 1]`.

| System | Label-0 F1 | P0 | R0 | Label-1 F1 | Macro F1 | Accuracy | Confusion | Predicted 0/1 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| all zero | 0.625287 | 0.454849 | 1.000000 | 0.000000 | 0.312644 | 0.454849 | `[[408,0],[489,0]]` | 897 / 0 |
| regenerated V4-A full | **0.705274** | 0.610018 | 0.835784 | 0.655381 | 0.680327 | 0.682274 | `[[341,67],[218,271]]` | 559 / 338 |
| substring + I | 0.653944 | 0.679894 | 0.629902 | 0.730159 | 0.692051 | 0.696767 | `[[257,151],[121,368]]` | 378 / 519 |
| substring + R | 0.653846 | 0.685484 | 0.625000 | 0.733728 | 0.693787 | 0.698997 | `[[255,153],[117,372]]` | 372 / 525 |
| substring + U | 0.656489 | 0.682540 | 0.632353 | 0.732143 | 0.694316 | 0.698997 | `[[258,150],[120,369]]` | 378 / 519 |
| V4-A present + I absent | 0.648787 | 0.677333 | 0.622549 | 0.727992 | 0.688389 | 0.693423 | `[[254,154],[121,368]]` | 375 / 522 |
| V4-A present + R absent | 0.648649 | 0.682927 | 0.617647 | 0.731563 | 0.690106 | 0.695652 | `[[252,156],[117,372]]` | 369 / 528 |
| V4-A present + U absent | 0.651341 | 0.680000 | 0.625000 | 0.729970 | 0.690656 | 0.695652 | `[[255,153],[120,369]]` | 375 / 522 |

The corrected ranking differs materially from the old macro-F1 ranking: regenerated V4-A at
0.50 is the strongest genuine fixed-threshold class-0-F1 baseline even though substring/U had
the strongest macro F1 among the routed fixed-threshold systems.

## Genuine nested V14 results

| Candidate | Label-0 F1 | P0 | R0 | Label-1 F1 | Macro F1 | Accuracy | Confusion | Predicted 0 share | Seed F1-0 (17/29/43) | Improved seeds/folds | 95% paired group-bootstrap difference |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---|
| Y0 | 0.725854 | 0.602917 | 0.911765 | 0.634590 | 0.680222 | 0.686734 | `[[372,36],[245,244]]` | 68.78% | 0.725146 / 0.725664 / 0.726744 | 3 / 12 | `[0.005634, 0.036795]` |
| Y1 | 0.733014 | 0.601256 | 0.938725 | 0.627503 | 0.680259 | 0.688963 | `[[383,25],[254,235]]` | 71.01% | 0.731429 / 0.737143 / 0.730435 | 3 / 11 | `[0.008149, 0.048945]` |
| Y2 | 0.727097 | 0.599364 | 0.924020 | 0.626156 | 0.676627 | 0.684504 | `[[377,31],[252,237]]` | 70.12% | 0.726225 / 0.736232 / 0.718841 | 3 / 11 | `[0.000675, 0.044328]` |
| Y3 | 0.731141 | 0.603834 | 0.926471 | 0.634211 | 0.682676 | 0.690078 | `[[378,30],[248,241]]` | 69.79% | 0.737463 / 0.740741 / 0.715116 | 3 / 11 | `[0.005235, 0.047819]` |
| Y4 | **0.737143** | 0.602804 | 0.948529 | 0.629032 | 0.683088 | 0.692308 | `[[387,21],[255,234]]` | 71.57% | 0.738636 / 0.735043 / 0.737752 | 3 / 12 | `[0.012230, 0.052827]` |

The bootstrap used 10,000 iterations and fixed seed 141400. Candidate Y4 improves over the
best genuine threshold-0.50 baseline by **0.031869 label-0 F1**, passes all three seed checks,
keeps both prediction shares below 90%, and meets the predeclared strong-improvement gate.
Its 15 inner-selected absent thresholds were 0.85 five times and 0.90 ten times; the median
deployment threshold is **0.90**. No macro-F1 increase was required.

The 15-fold threshold distributions were: Y0 global `{0.55:2, 0.60:2, 0.65:5,
0.70:5, 0.75:1}`; Y1 present `{0.40:3, 0.45:1, 0.50:2, 0.55:6, 0.60:1,
0.70:2}` and absent `{0.85:5, 0.90:10}`; Y2 used the same present distribution and
absent `{0.70:2, 0.75:1, 0.80:2, 0.85:3, 0.90:7}`; Y3 used the same present
distribution and absent `{0.65:1, 0.70:4, 0.80:4, 0.85:5, 0.90:1}`; Y4 absent
`{0.85:5, 0.90:10}`. Seed label-0 F1 population standard deviations were 0.000666,
0.002956, 0.007127, 0.011386, and 0.001529 for Y0 through Y4 respectively.

## Separate deadline-only teammate audit

`unverified_v4a_crossfold_threshold_audit` used artifact SHA-256
`78ee80a3f9944500090342170ccf60db7cbc793c7fbe6a17ad3e6a6906d51ea6`.
Its model-training membership remains unverified, so it was excluded from scientific V14
selection. Cross-fold label-0 F1 was 0.701220 versus 0.710280 at threshold 0.50, a change of
-0.009061; only one of five folds improved. The selected thresholds were 0.75, 0.75, 0.75,
0.55, and 0.50. The gate failed, so no secondary rethresholded V4-A submission was generated.

## Eligible output and resources

The authenticated test file retained 2,516 rows and route totals 1,361 present / 1,155 absent.
The scientific Y4 submission contains exactly `id,label`, ordered unique IDs, binary labels,
no missing values, and no index column. It predicts 1,791 label-0 rows and 725 label-1 rows
(71.18% label 0).

- submission SHA-256: `055a4b49802b53ff750855f682e805387c87b98ef3f39351cd093cf96e5ecff5`;
- run-summary SHA-256: `dde14408b19676725a3451e31fa61fd166cd663448d28215bb1d5e6808f16534`;
- total runtime: 655.303 seconds;
- peak process working set: 3,516,882,944 bytes;
- peak allocated VRAM: 2,530,798,080 bytes on the authenticated GPU;
- workers: four retrieval, two NLI DataLoader, one classifier process, one GPU model at a time.

No file was uploaded to Kaggle. V14 source, tests, and this document remain uncommitted and
unpushed as required.

## Public result and generalization verdict

The locked V14 Y4 submission later received a Kaggle public-leaderboard score of **0.666**.
Despite its genuine nested official-sample label-0 F1 of **0.737143**, the correction did not
generalize to a competitive public result. The prior V4-A submission remains the verified
public best at **0.685**. These public scores are post-evaluation observations only and were
not used to alter V14 models, thresholds, or predictions.
