# V7 Implementation Status

| Milestone / experiment | Status | Evidence or blocker |
|---|---|---|
| Repository audit | COMPLETE | ZIP hashes, Git ancestry, datasets, artifacts, environment, and immutable originals audited |
| M0 freeze/branch | COMPLETE | `feat/v7-compatible-hybrid` starts at packaged V4-A `c79a68f`; original branch unchanged |
| M1 routing/grouping/diagnostics | COMPLETE | 291 groups, fixed 3x5 outer folds, nested grouped inner folds, 33 mandatory tests |
| E0 common-fold V4-A | COMPLETE | 897 OOF rows, 3,588 cross-fits, 15 reload-verified models |
| E1 exact rule + Candidate I | COMPLETE | 897 OOF rows, 3,588 cross-fits, 15 reload-verified models |
| E3 deterministic route blend | COMPLETE | Five predeclared rules, nested selection, required diagnostics |
| M2 overall | COMPLETE | E0/E1/E3 regenerated on identical folds |
| E2 evidence-conditioned BanglaBERT | COMPLETE | Pinned offline checkpoint, accepted/rejected evidence states, 15 reload-verified classifiers |
| M3 overall | COMPLETE | Frozen encoder evaluation completed; rejected as noncompetitive ablation |
| E4 nested logistic stack | REJECTED BY GATE | Complete nested stack; macro F1 delta versus E0 is -0.006999 |
| M4 overall | COMPLETE | Fusion code/artifacts complete; hard-stop gate evaluated honestly |
| M5 inference implementation | COMPLETE | Offline V4-A fallback entry point and frozen config exist |
| M5 local test inference/submission | COMPLETE | Verified V4-A artifact submitted; Kaggle Complete, public score 0.685, no gain over V4-A |
| Optional E5-E7 | NOT STARTED | Hard stop prohibits larger searches after E4 gate failure |

No atomic-claim splitting, claim-level loss, XNLI, public-5K adaptation, 7B judge, or external API
is claimed.
