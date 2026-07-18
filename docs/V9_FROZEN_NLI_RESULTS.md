# V9 Frozen Multilingual NLI Results

Status: **complete — Candidates R, S, and T rejected under the frozen gates**.

The single `v9_frozen_nli_verification` execution used the authenticated 299-row official
sample, corrected 130/169 routing, corpus manifest
`af7991f07ff0de36eab50853b4bf623bf9aac4f289da124b806281c538ab10cf`, and fixed threshold
0.50. Each official row received genuine grouped OOF evaluation once per seed. No test data,
public labels, NLI fine-tuning, threshold tuning, or row-level persistence was used.

## Aggregate results

All confusion matrices and prediction counts pool three repeated OOF seeds, so the null route
contains 507 evaluations and the routed result contains 897 evaluations.

| Candidate | Null macro F1 | F1 label 0 / 1 | Accuracy | Null confusion | Null counts 0 / 1 | Brier | Routed macro F1 | Routed confusion | Routed counts 0 / 1 |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| R — mDeBERTa + retrieval | 0.538316 | 0.530120 / 0.546512 | 0.538462 | `[[132,135],[99,141]]` | 231 / 276 | 0.250742 | 0.693787 | `[[255,153],[117,372]]` | 372 / 525 |
| S — XLM-R + retrieval | 0.456570 | 0.432990 / 0.480151 | 0.457594 | `[[105,162],[113,127]]` | 218 / 289 | 0.276308 | 0.645850 | `[[228,180],[131,358]]` | 359 / 538 |
| T — dual NLI + retrieval + V5 numeric | 0.514746 | 0.509960 / 0.519531 | 0.514793 | `[[128,139],[107,133]]` | 235 / 272 | 0.290992 | 0.680549 | `[[251,157],[125,364]]` | 376 / 521 |

Probability mean/standard deviation was 0.506807/0.160135 for R, 0.507321/0.117485 for S,
and 0.494639/0.248890 for T. The frozen present route remained 0.900026 macro F1 with
confusion matrix `[[123,18],[18,231]]` over three seeds.

Null-route seed macro-F1 mean/std/min/max:

- R: 0.538124 / 0.025934 / 0.502104 / 0.562115.
- S: 0.454875 / 0.021773 / 0.426146 / 0.478834.
- T: 0.514647 / 0.019168 / 0.491106 / 0.538057.

Routed seed macro-F1 mean/std/min/max:

- R: 0.693707 / 0.015571 / 0.672149 / 0.708379.
- S: 0.645388 / 0.014278 / 0.627079 / 0.661918.
- T: 0.680549 / 0.010428 / 0.667687 / 0.693228.

Every candidate passed the 0.04 seed-stability and 90% class-collapse guards. R exceeded the
routed reference 0.692051 by 0.001736, but its null-route score was 0.003447 below the null
champion 0.541763. The predeclared gates therefore reject R. S and T are below both champion
references and are also rejected. No V9 candidate is selected.

## Fold-level null-route macro F1

| Seed | Fold | Train | Valid | R F1 | S F1 | T F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1 | 136 | 33 | 0.592593 | 0.538677 | 0.600186 |
| 17 | 2 | 138 | 31 | 0.418750 | 0.337607 | 0.387097 |
| 17 | 3 | 134 | 35 | 0.449959 | 0.370915 | 0.507858 |
| 17 | 4 | 133 | 36 | 0.649351 | 0.416216 | 0.636364 |
| 17 | 5 | 135 | 34 | 0.586806 | 0.439560 | 0.529412 |
| 29 | 1 | 137 | 32 | 0.590148 | 0.343109 | 0.498039 |
| 29 | 2 | 133 | 36 | 0.405193 | 0.540426 | 0.405193 |
| 29 | 3 | 129 | 40 | 0.413333 | 0.524703 | 0.517460 |
| 29 | 4 | 140 | 29 | 0.482759 | 0.344828 | 0.512019 |
| 29 | 5 | 137 | 32 | 0.583584 | 0.533333 | 0.611336 |
| 43 | 1 | 136 | 33 | 0.393382 | 0.301923 | 0.424242 |
| 43 | 2 | 137 | 32 | 0.655914 | 0.530792 | 0.455455 |
| 43 | 3 | 137 | 32 | 0.686275 | 0.530792 | 0.527094 |
| 43 | 4 | 136 | 33 | 0.467236 | 0.404558 | 0.467236 |
| 43 | 5 | 130 | 39 | 0.582888 | 0.484127 | 0.538158 |

All 15 folds had zero strengthened-group overlap, both labels in training and validation, exact
OOF coverage, training-only scaler/lexical fitted state, and recorded training/validation index
hashes. Feature configuration fingerprint:
`c7cec06dd5a1569fce041af4b50cddaa4c42681539c5a8c962571f622c404f26`.

## Frozen NLI diagnostics

| Model | Entailment mean / std | Contradiction mean / std | Neutral mean / std | Margin mean / std | Entailment-dominant rows | Retrieval/entailment correlation |
|---|---:|---:|---:|---:|---:|---:|
| mDeBERTa | 0.323809 / 0.278614 | 0.273677 / 0.259264 | 0.402514 / 0.317212 | 0.050132 / 0.434814 | 93.491% | 0.129627 |
| XLM-R | 0.105743 / 0.207710 | 0.143829 / 0.269481 | 0.750427 / 0.342314 | -0.038086 / 0.338154 | 96.450% | 0.062869 |

Both authenticated label mappings agreed with fixed synthetic entailment/contradiction behavior,
all probability vectors summed to one, and no NaN or Inf occurred. Each model scored exactly
845 pairs. mDeBERTa inference took 8.352 seconds and XLM-R took 8.215 seconds.

## Resources and safety

The project environment used Python 3.12.6 and pip 26.1.2. The prebuilt wheel
`sentencepiece-0.2.1-cp312-cp312-win_amd64.whl` installed with `--no-deps` only inside
`F:\datathon\olikbochon\.venv`. The imported version was exactly 0.2.1 from
`F:\datathon\olikbochon\.venv\Lib\site-packages\sentencepiece\__init__.py`; distribution
metadata was under `sentencepiece-0.2.1.dist-info`. Its 21-entry `RECORD` SHA-256 was
`da8f332b1c6c3ae602e3161734b45eee66a127bf4728224a798db81027c93c3a`. `pip check` passed,
and comparison with the pre-install freeze confirmed no other package changed.

Synthetic batches 16, 32, and 64 all remained below 85% VRAM, so both models selected batch 64.
Peak allocated VRAM was 2,530,798,080 bytes for mDeBERTa (19.65% of 12,878,086,144 bytes) and
1,703,157,248 bytes for XLM-R (13.23%). Models ran sequentially with two DataLoader workers and
pinned memory. Retrieval used four CPU threads with one corpus index. Classifiers used three
spawn workers and one BLAS thread each; RAM safety reduction was not triggered.

Wall time was 150.342 seconds. Peak main-process working set was 6,103,130,112 bytes and maximum
recorded system-memory load was 49%. No raw text, article title, row-level probability, model
weight, prediction, or submission was persisted in tracked files.
