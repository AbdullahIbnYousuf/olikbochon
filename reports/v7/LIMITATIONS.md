# Limitations

- Only 299 labeled rows are available; all reported uncertainty is fold/seed variability, not a
  large independent validation set.
- The public Wikimedia 2026-07-01 dump replaces the unavailable historical Kaggle `bnwiki`
  resource. The audited V4 loader uses the predeclared AA/AB/AC/AD subset: 400 shards,
  60,876 usable articles, and 826,876,458 bytes. This is compatible code execution, not a claim
  of byte identity with the historical corpus.
- E2 uses a frozen mean-pooled BanglaBERT encoder because no CUDA GPU was available. It is an
  authenticated semantic ablation, not full fine-tuning, atomic-claim verification, or NLI.
- Error categories lack category-level gold annotations; overlap statistics are reliable, but
  causal category attribution would require a declared annotation protocol.
- V7 did not improve on V4-A. E4 lost 0.006999 macro F1 against E0 on the frozen validation
  protocol, so the hard-stop promoted the exact preserved V4-A artifact. Its Kaggle public score
  is 0.685, identical to the prior V4-A score.
- Repository-wide V3/V4 notebook/vendor archive tests have three pre-existing Windows line-ending
  digest failures. The same failures reproduce on the untouched packaged V4-A head; all other
  183 tests pass.
- The one-time Wikipedia cache build took 1,754 seconds on this CPU. Fold evaluation and frozen
  E2 embedding/classification are far below the nine-hour target. The submitted artifact was the
  preserved V4-A fallback; a fresh end-to-end P100/T4 rerun was not needed for that exact artifact.
