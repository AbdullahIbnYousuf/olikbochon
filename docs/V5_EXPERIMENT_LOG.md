# Version 5 Experiment Log

## `v5_official_lexical_baseline`

Status: complete. This is the only V5 grouped lexical run.

- data: authenticated official 299-row labeled sample only;
- route: 130 context-present and 169 context-absent;
- validation: seeds 17/29/43, five strengthened grouped folds per seed;
- grouping: 291 groups, largest group 2, zero train/validation overlap in all 15 folds;
- route feasibility: both labels in every route-specific train and validation split;
- rare-token frequencies: fitted independently on each route's training fold;
- neural inference/downloads/test access: none;
- wall time: approximately 15 seconds on CPU;
- artifacts: two ignored JSON files, 379,114 bytes total;
- artifact audit: no raw-value matches, no raw-text substring matches of length at least 8,
  no row-level table, and no unexpected file types.

All pooled confusion matrices and prediction counts below contain three repeated OOF copies of
the official sample, one per seed. They are not 897 independent examples.

## Aggregate feature statistics

The full predeclared count/mean/median/population-SD/minimum/maximum table for all 48 route
features is in `V5_MODELING_PLAN.md` and is also preserved under `feature_audit` in the ignored
aggregate result. Key means are:

| Route/feature | Label 0 | Label 1 |
|---|---:|---:|
| present: exact response substring | 0.127660 | 0.927711 |
| present: punctuation-stripped substring | 0.148936 | 0.927711 |
| present: response-token coverage | 0.589805 | 0.925703 |
| present: longest-common-substring ratio | 0.512386 | 0.948998 |
| present: numeric consistency | 0.797872 | 0.975904 |
| present: response characters | 35.170213 | 13.240964 |
| absent: response characters | 15.056180 | 13.362500 |
| absent: prompt-response char-3 Jaccard | 0.031071 | 0.036496 |
| absent: Bengali-script ratio | 0.769101 | 0.887500 |
| absent: specificity score | 2.988764 | 2.825000 |

## Aggregate candidate results

| Candidate | Threshold | Macro F1 | F1-0 | F1-1 | Accuracy | Confusion matrix | Predicted 0/1 | Brier | Probability mean/SD | Guard |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| A routed rule | frozen | 0.682326 | 0.738636 | 0.626016 | 0.692308 | `[[390,18],[258,231]]` | 648/249 | n/a | n/a | n/a |
| B present logistic | 0.50 | 0.871032 | 0.833333 | 0.908730 | 0.882051 | `[[115,26],[20,229]]` | 135/255 | 0.091501 | 0.586764/0.388694 | pass |
| B present logistic | **0.30 selected** | **0.888955** | 0.853933 | 0.923977 | 0.900000 | `[[114,27],[12,237]]` | 126/264 | 0.091501 | 0.586764/0.388694 | pass |
| C absent logistic | 0.50 selected | 0.541763 | 0.524590 | 0.558935 | 0.542406 | `[[128,139],[93,147]]` | 221/286 | 0.273965 | 0.490229/0.186593 | pass |
| D routed logistic | **0.50 selected** | **0.683111** | 0.636126 | 0.730097 | 0.690078 | `[[243,165],[113,376]]` | 356/541 | 0.194633 | 0.532201/0.296070 | pass |

Candidate A context-present metrics are macro F1 0.900026, F1-0 0.872340, F1-1
0.927711, accuracy 0.907692, confusion matrix `[[123,18],[18,231]]`, and predicted counts
141/249. Its context-absent fallback was label 0 for every fold: seeds 17, 29, and 43,
folds 1–5. The absent-route macro F1 was 0.344961; the resulting aggregate routed score is
0.682326.

At Candidate D's selected threshold 0.50, context-present macro F1 is 0.871032 and
context-absent macro F1 is 0.541763. Threshold gain over 0.50 is exactly 0.000000. Its
selected-threshold per-seed macro-F1 mean is 0.683024, population SD 0.007108, minimum
0.673134, and maximum 0.689526. Maximum predicted-class share is 0.603122, so the collapse
guard passes.

## Per-seed results

| Seed | Candidate/threshold | Macro F1 | F1-0 | F1-1 | Accuracy | Confusion matrix | Predicted 0/1 |
|---:|---|---:|---:|---:|---:|---|---:|
| 17 | A/frozen | 0.682326 | 0.738636 | 0.626016 | 0.692308 | `[[130,6],[86,77]]` | 216/83 |
| 17 | B/0.50 | 0.866701 | 0.829787 | 0.903614 | 0.876923 | `[[39,8],[8,75]]` | 47/83 |
| 17 | B/0.42 selected | 0.882246 | 0.847826 | 0.916667 | 0.892308 | `[[39,8],[6,77]]` | 45/85 |
| 17 | C/0.50 | 0.555653 | 0.539877 | 0.571429 | 0.556213 | `[[44,45],[30,50]]` | 74/95 |
| 17 | C/0.48 selected | 0.565063 | 0.529032 | 0.601093 | 0.568047 | `[[41,48],[25,55]]` | 66/103 |
| 17 | D/0.50 | 0.689526 | 0.645914 | 0.733138 | 0.695652 | `[[83,53],[38,125]]` | 121/178 |
| 17 | D/0.48 selected | 0.696866 | 0.645161 | 0.748571 | 0.705686 | `[[80,56],[32,131]]` | 112/187 |
| 29 | A/frozen | 0.682326 | 0.738636 | 0.626016 | 0.692308 | `[[130,6],[86,77]]` | 216/83 |
| 29 | B/0.50 | 0.873204 | 0.835165 | 0.911243 | 0.884615 | `[[38,9],[6,77]]` | 44/86 |
| 29 | B/0.30 selected | 0.896934 | 0.863636 | 0.930233 | 0.907692 | `[[38,9],[3,80]]` | 41/89 |
| 29 | C/0.50 selected | 0.524613 | 0.493671 | 0.555556 | 0.526627 | `[[39,50],[30,50]]` | 69/100 |
| 29 | D/0.50 selected | 0.673134 | 0.618474 | 0.727794 | 0.682274 | `[[77,59],[36,127]]` | 113/186 |
| 43 | A/frozen | 0.682326 | 0.738636 | 0.626016 | 0.692308 | `[[130,6],[86,77]]` | 216/83 |
| 43 | B/0.50 | 0.873204 | 0.835165 | 0.911243 | 0.884615 | `[[38,9],[6,77]]` | 44/86 |
| 43 | B/0.34 selected | 0.887715 | 0.850575 | 0.924855 | 0.900000 | `[[37,10],[3,80]]` | 40/90 |
| 43 | C/0.50 | 0.544315 | 0.538922 | 0.549708 | 0.544379 | `[[45,44],[33,47]]` | 78/91 |
| 43 | C/0.52 selected | 0.549902 | 0.563218 | 0.536585 | 0.550296 | `[[49,40],[36,44]]` | 85/84 |
| 43 | D/0.50 | 0.686411 | 0.643411 | 0.729412 | 0.692308 | `[[83,53],[39,124]]` | 122/177 |
| 43 | D/0.52 selected | 0.688513 | 0.654135 | 0.722892 | 0.692308 | `[[87,49],[43,120]]` | 130/169 |

Per-seed calibration for B is: seed 17 Brier/mean/SD 0.096257/0.570936/0.393871;
seed 29 0.089623/0.594834/0.387025; seed 43 0.088624/0.594521/0.384643.
For C it is: seed 17 0.267506/0.493837/0.178220; seed 29
0.274551/0.493057/0.192580; seed 43 0.279838/0.483794/0.188519. Every selected
seed threshold passes the collapse guard.

## Per-fold Candidates B and C

Each metric cell is `macro F1 / F1-0 / F1-1 / accuracy`. Probability cells are
`mean / population SD`. Every selected fold threshold passed the collapse guard.

### Candidate B: context-present logistic

| Seed/fold | Metrics @0.50 | CM | Pred 0/1 | Selected threshold/macro | Brier | Probability mean/SD |
|---|---|---|---:|---:|---:|---:|
| 17/1 | 0.813793/0.800000/0.827586/0.814815 | `[[10,2],[3,12]]` | 13/14 | 0.42/0.850000 | 0.147371 | 0.491133/0.431696 |
| 17/2 | 0.856459/0.818182/0.894737/0.866667 | `[[9,3],[1,17]]` | 10/20 | 0.48/0.890110 | 0.089271 | 0.598226/0.382184 |
| 17/3 | 0.913194/0.888889/0.937500/0.920000 | `[[8,0],[2,15]]` | 10/15 | 0.50/0.913194 | 0.092486 | 0.520171/0.384201 |
| 17/4 | 0.940260/0.909091/0.971429/0.956522 | `[[5,0],[1,17]]` | 6/17 | 0.50/0.940260 | 0.045576 | 0.655942/0.356726 |
| 17/5 | 0.826389/0.777778/0.875000/0.840000 | `[[7,3],[1,14]]` | 8/17 | 0.80/0.918831 | 0.099833 | 0.596934/0.384442 |
| 29/1 | 0.844444/0.800000/0.888889/0.857143 | `[[8,1],[3,16]]` | 11/17 | 0.32/0.918129 | 0.083675 | 0.548558/0.380129 |
| 29/2 | 0.844075/0.769231/0.918919/0.880000 | `[[5,1],[2,17]]` | 7/18 | 0.30/0.890351 | 0.118259 | 0.673974/0.347029 |
| 29/3 | 0.946779/0.941176/0.952381/0.947368 | `[[8,0],[1,10]]` | 9/10 | 0.50/0.946779 | 0.065171 | 0.464755/0.426689 |
| 29/4 | 0.852381/0.800000/0.904762/0.870968 | `[[8,4],[0,19]]` | 8/23 | 0.56/0.859091 | 0.098089 | 0.660414/0.378426 |
| 29/5 | 0.883117/0.857143/0.909091/0.888889 | `[[9,3],[0,15]]` | 9/18 | 0.56/0.923295 | 0.076763 | 0.585790/0.377302 |
| 43/1 | 1.000000/1.000000/1.000000/1.000000 | `[[8,0],[0,19]]` | 8/19 | 0.50/1.000000 | 0.040951 | 0.610755/0.347235 |
| 43/2 | 0.891925/0.869565/0.914286/0.896552 | `[[10,1],[2,16]]` | 12/17 | 0.34/0.926768 | 0.092027 | 0.528674/0.415684 |
| 43/3 | 0.754386/0.666667/0.842105/0.785714 | `[[6,4],[2,16]]` | 8/20 | 0.64/0.844444 | 0.144253 | 0.666739/0.361755 |
| 43/4 | 0.837500/0.800000/0.875000/0.846154 | `[[8,3],[1,14]]` | 9/17 | 0.72/0.883058 | 0.084953 | 0.579659/0.396916 |
| 43/5 | 0.890110/0.857143/0.923077/0.900000 | `[[6,1],[1,12]]` | 7/13 | 0.34/0.943020 | 0.074937 | 0.586302/0.382148 |

### Candidate C: context-absent logistic

| Seed/fold | Metrics @0.50 | CM | Pred 0/1 | Selected threshold/macro | Brier | Probability mean/SD |
|---|---|---|---:|---:|---:|---:|
| 17/1 | 0.575368/0.562500/0.588235/0.575758 | `[[9,6],[8,10]]` | 17/16 | 0.48/0.689850 | 0.250384 | 0.478419/0.089514 |
| 17/2 | 0.643678/0.620690/0.666667/0.645161 | `[[9,7],[4,11]]` | 13/18 | 0.50/0.643678 | 0.261869 | 0.501782/0.165723 |
| 17/3 | 0.512695/0.484848/0.540541/0.514286 | `[[8,11],[6,10]]` | 14/21 | 0.52/0.542484 | 0.276242 | 0.509329/0.178744 |
| 17/4 | 0.609907/0.631579/0.588235/0.611111 | `[[12,10],[4,10]]` | 16/20 | 0.58/0.674074 | 0.256466 | 0.471197/0.231185 |
| 17/5 | 0.436792/0.387097/0.486486/0.441176 | `[[6,11],[8,9]]` | 14/20 | 0.66/0.525581 | 0.291960 | 0.509584/0.185081 |
| 29/1 | 0.437500/0.437500/0.437500/0.437500 | `[[7,11],[7,7]]` | 14/18 | 0.42/0.583584 | 0.258327 | 0.447982/0.218019 |
| 29/2 | 0.471815/0.457143/0.486486/0.472222 | `[[8,14],[5,9]]` | 13/23 | 0.60/0.542857 | 0.303928 | 0.526865/0.197240 |
| 29/3 | 0.545455/0.500000/0.590909/0.550000 | `[[9,10],[8,13]]` | 17/23 | 0.54/0.548872 | 0.275862 | 0.511316/0.195145 |
| 29/4 | 0.551724/0.551724/0.551724/0.551724 | `[[8,7],[6,8]]` | 14/15 | 0.50/0.551724 | 0.278118 | 0.441193/0.183207 |
| 29/5 | 0.611336/0.538462/0.684211/0.625000 | `[[7,8],[4,13]]` | 11/21 | 0.48/0.638974 | 0.252855 | 0.524274/0.139285 |
| 43/1 | 0.514706/0.529412/0.500000/0.515152 | `[[9,10],[6,8]]` | 15/18 | 0.52/0.572222 | 0.280085 | 0.498327/0.141462 |
| 43/2 | 0.468231/0.451613/0.484848/0.468750 | `[[7,10],[7,8]]` | 14/18 | 0.54/0.468231 | 0.326413 | 0.491351/0.200860 |
| 43/3 | 0.647648/0.702703/0.592593/0.656250 | `[[13,4],[7,8]]` | 20/12 | 0.50/0.647648 | 0.248108 | 0.440233/0.190104 |
| 43/4 | 0.477167/0.413793/0.540541/0.484848 | `[[6,10],[7,10]]` | 13/20 | 0.42/0.529915 | 0.295513 | 0.493256/0.215787 |
| 43/5 | 0.587302/0.555556/0.619048/0.589744 | `[[10,10],[6,13]]` | 16/23 | 0.52/0.615385 | 0.254186 | 0.493033/0.181577 |

## Standardized coefficient summaries

All 30 complete per-seed/fold coefficient vectors are preserved in the ignored aggregate
result under `fold_coefficients`. The following are the five largest mean directions in each
route. All listed signs were stable across all 15 fits.

| Route | Direction | Feature | Mean coefficient | SD |
|---|---|---|---:|---:|
| present | positive | response_in_context_exact | 1.365023 | 0.189852 |
| present | positive | response_in_context_no_punctuation | 0.680788 | 0.218524 |
| present | positive | prompt_char_count | 0.577183 | 0.136895 |
| present | positive | numeric_consistency_ratio | 0.515698 | 0.093005 |
| present | positive | char4_jaccard | 0.399457 | 0.089252 |
| present | negative | response_char_count | -0.886984 | 0.178252 |
| present | negative | response_token_coverage | -0.653971 | 0.176291 |
| present | negative | response_token_count | -0.640245 | 0.191516 |
| present | negative | prompt_token_count | -0.514380 | 0.117488 |
| present | negative | response_character_coverage | -0.511065 | 0.236308 |
| absent | positive | response_token_count | 0.973592 | 0.266264 |
| absent | positive | response_arabic_numeral_count | 0.379832 | 0.154000 |
| absent | positive | prompt_response_char3_jaccard | 0.325287 | 0.116109 |
| absent | positive | response_numeral_count | 0.166661 | 0.071862 |
| absent | negative | response_char_count | -0.563413 | 0.140193 |
| absent | negative | response_negation_count | -0.554250 | 0.065686 |
| absent | negative | prompt_response_token_jaccard | -0.370492 | 0.134313 |
| absent | negative | response_named_entity_like_count | -0.361818 | 0.250442 |
| absent | negative | response_punctuation_count | -0.339518 | 0.118121 |

The present-model negative conditional coefficients for coverage features coexist with strong
positive exact-match coefficients and should not be interpreted causally; the frozen lexical
features are correlated. Features whose full sign-stability records are not stable must not be
overinterpreted.

## Acceptance verdicts

- Candidate A: **strong**. Context-present macro F1 0.900026 exceeds both the 0.75 promising
  and 0.85 strong gates.
- Candidate B: **does not pass the incremental gate**. Its selected macro F1 0.888955 trails
  Candidate A by 0.011071. Its dominant-class share is 0.676923 versus Candidate A's 0.638462,
  so it does not materially improve class balance.
- Candidate C: **weak / fail**. Selected macro F1 0.541763 is 0.008237 below 0.55.
- Candidate D: **high-value**. Selected macro F1 0.683111 is 0.118015 above the corrected
  BanglaBERT baseline 0.565096, 0.063111 above the strong gate 0.62, and 0.003111 above the
  high-value gate 0.68. It is not rejected.

No acceptance rule, feature, threshold grid, model, hyperparameter, route, or fallback was
changed after results were observed.
