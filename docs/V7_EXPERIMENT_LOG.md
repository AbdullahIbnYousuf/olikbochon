# V7 null-route retrieval experiment log

Status: **complete — all frozen candidates rejected**. Candidate N is the best V7
candidate, but it does not replace either the null-route or routed champion. This log
contains aggregate and fold-level numeric metadata only; no article, prompt, response,
evidence, vocabulary, probability, prediction, or competition-test content is persisted.

## Authenticated inputs and frozen configuration

- official labeled sample: 299 rows; corrected routes 130 present / 169 absent;
- corpus: 602 physical chunks, 301 unique contents, 301 exact duplicate copies;
- parsed / usable records: 65,256 / 62,153;
- logical corpus manifest SHA-256:
  `af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf`;
- corpus index: `char_wb` character TF-IDF, n-grams 2–4, 50,000 maximum features,
  float32, L2 normalization;
- index dimensions: 62,153 × 50,000; 96,596,030 nonzero values; sparse-array bytes
  773,016,856;
- Candidate M: top-1 intro evidence and the frozen eight V4-A evidence features;
- Candidate N: top-5 query-centered evidence and seven frozen aggregate features;
- Candidate P: the eight evidence-feature schema and seven aggregate features over the
  approved top-5 query-centered evidence, plus 21 train-fold-only V5 null features;
- classifier: standardized balanced logistic regression, C=1, liblinear, 3,000 maximum
  iterations, random state 42;
- outer validation: seeds 17, 29, and 43 × five strengthened grouped folds;
- inner selection: three grouped folds inside each outer-training partition; cutoff grid
  0.05–0.40 by 0.05 and classifier threshold grid 0.20–0.80 by 0.02;
- group overlap: zero in all outer and inner splits; every required split contained both
  labels; official queries used to fit the corpus vocabulary: zero.

## Aggregate results

All confusion matrices and prediction counts aggregate the three complete OOF passes
(507 null-route decisions or 897 routed decisions).

| Candidate | Evaluation | Macro F1 | F1 label 0 | F1 label 1 | Accuracy | Confusion matrix | Predicted 0/1 |
|---|---|---:|---:|---:|---:|---|---:|
| M | null @ 0.50 | 0.501093 | 0.531599 | 0.470588 | 0.502959 | `[[143,124],[128,112]]` | 271/236 |
| M | null nested | 0.491069 | 0.523191 | 0.458947 | 0.493097 | `[[141,126],[131,109]]` | 272/235 |
| M | routed @ 0.50 | 0.676546 | 0.648780 | 0.704312 | 0.678930 | `[[266,142],[146,343]]` | 412/485 |
| M | routed nested | 0.670994 | 0.643118 | 0.698869 | 0.673356 | `[[264,144],[149,340]]` | 413/484 |
| N | null @ 0.50 | 0.498732 | 0.532348 | 0.465116 | 0.500986 | `[[144,123],[130,110]]` | 274/233 |
| N | null nested | **0.515671** | 0.573944 | 0.457399 | 0.522682 | `[[163,104],[138,102]]` | 301/206 |
| N | routed @ 0.50 | 0.675607 | 0.648846 | 0.702369 | 0.677815 | `[[267,141],[148,341]]` | 415/482 |
| N | routed nested | **0.689225** | 0.672941 | 0.705508 | 0.690078 | `[[286,122],[156,333]]` | 442/455 |
| P | null @ 0.50 | 0.482991 | 0.471774 | 0.494208 | 0.483235 | `[[117,150],[112,128]]` | 229/278 |
| P | null nested | 0.463158 | 0.449393 | 0.476923 | 0.463511 | `[[111,156],[116,124]]` | 227/280 |
| P | routed @ 0.50 | 0.661830 | 0.616967 | 0.706693 | 0.667781 | `[[240,168],[130,359]]` | 370/527 |
| P | routed nested | 0.650269 | 0.603093 | 0.697446 | 0.656633 | `[[234,174],[134,355]]` | 368/529 |

The context-present frozen substring rule remained 0.900026 macro F1 for every routed
candidate. Nested null-route calibration statistics were:

| Candidate | Brier | Probability mean/std | Seed macro F1 mean/std/min/max | Maximum class share | Collapse guard |
|---|---:|---:|---:|---:|---|
| M | 0.264712 | 0.498989 / 0.111560 | 0.489192 / 0.018075 / 0.465190 / 0.508807 | 0.536489 | pass |
| N | 0.264060 | 0.507406 / 0.111374 | 0.515007 / 0.029058 / 0.477895 / 0.548846 | 0.593688 | pass |
| P | 0.310872 | 0.496033 / 0.245798 | 0.462120 / 0.020532 / 0.443300 / 0.490678 | 0.552268 | pass |

Routed seed macro F1 mean/std/min/max was 0.670614/0.010021/0.656461/0.678322
for M, 0.689036/0.018134/0.667945/0.712217 for N, and
0.649982/0.013316/0.638691/0.668679 for P.

## Nested outer-fold selections

The macro F1 column is the null-route score using the cutoff and threshold selected only
from the current outer-training partition's inner grouped OOF predictions.

| Candidate | Seed | Fold | Cutoff | Threshold | Macro F1 | Accepted | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|
| M | 17 | 1 | 0.20 | 0.52 | 0.415657 | 32 | 1 |
| M | 17 | 2 | 0.25 | 0.48 | 0.413866 | 25 | 6 |
| M | 17 | 3 | 0.30 | 0.50 | 0.570025 | 25 | 10 |
| M | 17 | 4 | 0.20 | 0.48 | 0.542857 | 36 | 0 |
| M | 17 | 5 | 0.20 | 0.54 | 0.350694 | 31 | 3 |
| M | 29 | 1 | 0.25 | 0.50 | 0.445887 | 25 | 7 |
| M | 29 | 2 | 0.30 | 0.46 | 0.555556 | 20 | 16 |
| M | 29 | 3 | 0.15 | 0.48 | 0.448622 | 40 | 0 |
| M | 29 | 4 | 0.30 | 0.48 | 0.480287 | 22 | 7 |
| M | 29 | 5 | 0.20 | 0.48 | 0.533333 | 28 | 4 |
| M | 43 | 1 | 0.20 | 0.52 | 0.503759 | 33 | 0 |
| M | 43 | 2 | 0.20 | 0.50 | 0.546559 | 30 | 2 |
| M | 43 | 3 | 0.30 | 0.50 | 0.417004 | 19 | 13 |
| M | 43 | 4 | 0.30 | 0.58 | 0.482949 | 20 | 13 |
| M | 43 | 5 | 0.20 | 0.50 | 0.507641 | 36 | 3 |
| N | 17 | 1 | 0.30 | 0.54 | 0.538677 | 20 | 13 |
| N | 17 | 2 | 0.25 | 0.46 | 0.388158 | 25 | 6 |
| N | 17 | 3 | 0.30 | 0.54 | 0.510490 | 25 | 10 |
| N | 17 | 4 | 0.40 | 0.50 | 0.584738 | 16 | 20 |
| N | 17 | 5 | 0.30 | 0.56 | 0.613636 | 21 | 13 |
| N | 29 | 1 | 0.10 | 0.50 | 0.405670 | 32 | 0 |
| N | 29 | 2 | 0.30 | 0.54 | 0.495798 | 20 | 16 |
| N | 29 | 3 | 0.40 | 0.50 | 0.386189 | 17 | 23 |
| N | 29 | 4 | 0.30 | 0.56 | 0.480287 | 22 | 7 |
| N | 29 | 5 | 0.30 | 0.52 | 0.560784 | 23 | 9 |
| N | 43 | 1 | 0.10 | 0.50 | 0.259512 | 33 | 0 |
| N | 43 | 2 | 0.30 | 0.52 | 0.611336 | 25 | 7 |
| N | 43 | 3 | 0.30 | 0.46 | 0.530792 | 19 | 13 |
| N | 43 | 4 | 0.30 | 0.54 | 0.503759 | 20 | 13 |
| N | 43 | 5 | 0.30 | 0.54 | 0.598490 | 24 | 15 |
| P | 17 | 1 | 0.35 | 0.56 | 0.450000 | 17 | 16 |
| P | 17 | 2 | 0.15 | 0.50 | 0.507937 | 31 | 0 |
| P | 17 | 3 | 0.30 | 0.50 | 0.475000 | 25 | 10 |
| P | 17 | 4 | 0.15 | 0.34 | 0.442724 | 36 | 0 |
| P | 17 | 5 | 0.15 | 0.38 | 0.368700 | 34 | 0 |
| P | 29 | 1 | 0.10 | 0.64 | 0.345455 | 32 | 0 |
| P | 29 | 2 | 0.30 | 0.50 | 0.442724 | 20 | 16 |
| P | 29 | 3 | 0.15 | 0.46 | 0.466667 | 40 | 0 |
| P | 29 | 4 | 0.30 | 0.48 | 0.482759 | 22 | 7 |
| P | 29 | 5 | 0.30 | 0.48 | 0.619048 | 23 | 9 |
| P | 43 | 1 | 0.10 | 0.54 | 0.361290 | 33 | 0 |
| P | 43 | 2 | 0.20 | 0.46 | 0.343109 | 30 | 2 |
| P | 43 | 3 | 0.25 | 0.46 | 0.500000 | 22 | 10 |
| P | 43 | 4 | 0.30 | 0.48 | 0.450000 | 20 | 13 |
| P | 43 | 5 | 0.15 | 0.48 | 0.538158 | 39 | 0 |

Cutoff counts were M `{0.15:1, 0.20:7, 0.25:2, 0.30:5}`, N
`{0.10:2, 0.25:1, 0.30:10, 0.40:2}`, and P
`{0.10:2, 0.15:5, 0.20:1, 0.25:1, 0.30:5, 0.35:1}`. N has a defensible
0.30 concentration; P is comparatively unstable.

## Retrieval diagnostics

Across the 169 unique null-route queries, top-1 score mean/median/std was
0.364342/0.350672/0.117011. The top-1 minus top-2 gap mean/median/std was
0.043687/0.024950/0.049100 (range 0.000063–0.281295).

| Candidate | Coverage | Accepted/rejected | Accepted-count distribution | No-evidence % | Accepted F1 | Rejected/no-evidence F1 | Runtime s |
|---|---:|---:|---|---:|---:|---:|---:|
| M | 0.832347 | 422/85 | `{0:85,1:422}` | 16.765286 | 0.478099 | 0.544269 | 31.497 |
| N | 0.674556 | 342/165 | `{0:165,1:59,2:27,3:20,4:14,5:222}` | 32.544379 | 0.509848 | 0.396371 | 154.473 |
| P | 0.836292 | 424/83 | `{0:83,1:33,2:13,3:16,4:10,5:352}` | 16.370809 | 0.455804 | 0.484929 | 195.491 |

Aggregate support-feature means by true label 0/1 are shown below. Standard deviations
are retained in the ignored aggregate results artifact.

| Candidate | Feature | Label 0 mean | Label 1 mean |
|---|---|---:|---:|
| M | evidence accepted | 0.820225 | 0.845833 |
| M | word overlap ratio | 0.102559 | 0.131517 |
| M | numbers supported | 0.786517 | 0.787500 |
| M | number overlap ratio | 0.792135 | 0.800000 |
| M | response has numbers | 0.224719 | 0.225000 |
| M | response substring | 0.048689 | 0.066667 |
| M | response length | 15.056180 | 13.362500 |
| M | evidence length | 527.981273 | 573.420833 |
| N | maximum token coverage | 0.154806 | 0.218651 |
| N | maximum substring support | 0.078652 | 0.112500 |
| N | maximum numeric consistency | 0.538077 | 0.594444 |
| N | maximum character similarity | 0.024417 | 0.016968 |
| N | mean top-3 retrieval score | 0.326699 | 0.330327 |
| N | rank-1/rank-2 gap | 0.043334 | 0.044080 |
| N | accepted passage count | 2.752809 | 2.516667 |
| P | evidence accepted | 0.831461 | 0.841667 |
| P | word overlap ratio | 0.113983 | 0.134543 |
| P | numbers supported | 0.786517 | 0.800000 |
| P | number overlap ratio | 0.797753 | 0.818750 |
| P | response has numbers | 0.224719 | 0.225000 |
| P | response substring | 0.056180 | 0.062500 |
| P | response length | 15.056180 | 13.362500 |
| P | evidence length | 531.880150 | 567.741667 |
| P | maximum token coverage | 0.196504 | 0.252598 |
| P | maximum substring support | 0.104869 | 0.137500 |
| P | maximum numeric consistency | 0.692884 | 0.712500 |
| P | maximum character similarity | 0.034268 | 0.024947 |
| P | mean top-3 retrieval score | 0.326699 | 0.330327 |
| P | rank-1/rank-2 gap | 0.043334 | 0.044080 |
| P | accepted passage count | 3.872659 | 3.637500 |

## Frozen-gate verdicts

- **M rejected:** null 0.491069 is 0.050694 below 0.541763; routed 0.670994 is
  0.021057 below 0.692051; accepted retrievals underperform rejected/no-evidence rows.
- **N rejected, but best V7 candidate:** null 0.515671 is 0.026092 below 0.541763;
  routed 0.689225 is 0.002826 below 0.692051. Accepted retrievals outperform rejected
  rows, seed standard deviation is below 0.04, class balance passes, and cutoffs concentrate
  at 0.30, but the primary champion gates are not met.
- **P rejected:** null 0.463158 is 0.078605 below 0.541763; routed 0.650269 is
  0.041782 below 0.692051; accepted retrievals underperform rejected/no-evidence rows and
  cutoff selection is dispersed.

Against reproduced V4-A (0.433971 null / 0.687270 full), nested M changes by
+0.057098/-0.016276, N by +0.081700/+0.001955, and P by +0.029187/-0.037001.
None reaches the promising 0.58 null gate, the strong 0.70 routed gate, or either current
champion. Retrieval verification therefore stops without Candidate O/NLI.

## Resource and safety audit

- corpus load: 36.421 seconds;
- one shared index build: 75.534 seconds;
- total wall time: 550.266 seconds;
- maximum process working set: 2,980,225,024 bytes (2.98 GB decimal), below the 24 GB
  stop limit;
- persisted ignored artifacts: three aggregate JSON files totaling 181,566 bytes;
- candidate evaluations: 45 completed outer folds, with no retries;
- raw text, row-level values, retrieval index, fitted vocabulary, models, predictions, and
  submissions persisted: none;
- competition test, public labeled datasets, external APIs, and Candidate O/NLI used: no.
