# V12 deadline QA verifier

V12 is the final frozen development cycle. It treats context-absent rows as Bengali short-answer
verification rather than stylistic hallucination detection. It evaluates exactly two Qwen3-4B
judge modes on the locked 127-row discovery partition, freezes one without holdout access, and
uses the 42-row locked holdout once. Competition inference is automatically gated on the frozen
holdout criteria.

The local rulebook permits publicly available external data and open-weight pretrained models,
provided the external data is declared/cited, contains no competition-test derivative, and all
inference is local. The public demonstration pool therefore uses the authenticated 4,000-row
train and 1,000-row validation files from `abidur14004/new-dataset`. Its 5,000-row contrastive
file conflicts with the historical 1,000-row expectation and is authenticated but excluded.

The immutable model is `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`, Apache-2.0, safetensors only. Its 11-file local
manifest is `3e5901f633796ce01cddff36c8136a3d80095147e7a0126c66ce54d716ff1251`
and totals 8,060,908,199 bytes. It loads offline with `trust_remote_code=False`, BF16,
`device_map="cuda"`, and SDPA. `accelerate==1.14.0` and `psutil==7.2.2` were added to the
project-local environment solely to support Transformers device mapping; no existing dependency
was upgraded.

Retrieval uses the frozen 0.65 character plus 0.35 word cosine, exactly three demonstrations per
label, deterministic score/index ordering, self/group exclusion, and only discovery/public labels.
Only unanimous exact normalized prompt-response matches override Qwen. AA is deterministic
non-thinking generation; AB is the fixed three-seed thinking majority. Both use threshold-free
hard JSON labels. No generated reasoning is persisted.

The winner is selected by discovery macro F1, with AA preferred within 0.01, maximum class share
0.85, and maximum invalid-output rate 0.02. Holdout passage requires macro F1 at least 0.70,
30/42 correct, improvement over both frozen Candidate I and R OOF baselines, class share at most
0.85, and no parsing/provenance failure. Only then may the authenticated competition test be
inferred and the final submission artifacts be created.

## Execution outcome

The first execution attempt exposed a self-match provenance defect in the exact-template lookup:
although the six retrieved demonstrations excluded the official target and its group, the separate
exact-match map did not. Its apparent 100% discovery template coverage and candidate selection
were invalid. No submission was generated. The ignored diagnostic was preserved under
`artifacts/v12/deadline_qa_verifier_invalid_self_match/`, and a focused regression test now proves
that self and same-group rows cannot provide an override.

The corrected execution ran the unchanged AA and AB definitions for 3,363.2 seconds. Both failed
the frozen discovery acceptance guards, so the runner stopped before constructing holdout targets
or opening the competition test. No locked-holdout metric, test prediction, emergency fallback, or
submission was generated. The fail-fast path did not persist the in-memory discovery aggregates;
recovering them would require rerunning the rejected candidates, which is outside the final-cycle
authorization. The safety verdict is therefore definitive, while the detailed AA/AB discovery
metrics are unavailable. No guard was weakened and no additional model cycle was run. Qwen
development is stopped: there is no valid V12 winner, test prediction, or submission.
