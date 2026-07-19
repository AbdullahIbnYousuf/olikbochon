# V13 public template transfer

V13 is the final development cycle. It uses the rule-permitted public resource
`abidur14004/new-dataset` and the authenticated local BanglaBERT snapshot. It does not run Qwen,
NLI, or another transformer; use competition-test labels; inspect test rows manually; or tune from
leaderboard feedback.

The 4,000-row train file and 1,000-row validation file retain their frozen responsibilities. The
unexpected contrastive file is authenticated as 5,000 balanced rows with the same four-column
schema and label orientation. It has no ID or test-label mapping column. Its local timestamps are
only weak provenance signals; the known Kaggle identifier, prior event-scoped approval, compatible
schema, and contamination audit are the primary admission evidence. Unknown upstream license and
citation metadata remain declared limitations.

Normalization is NFKC, Bengali/ASCII digit canonicalization, Latin lowercase, punctuation and
whitespace normalization, while retaining negation, entities, and semantic words. Skeletons replace
only numbers, dates, URLs, and obvious list numbering. Public rows matching validation pairs are
removed from training. Mixed-label pair conflicts are removed. For official diagnostics, all public
rows matching any official pair or skeleton are conservatively excluded before fitting every
candidate.

Candidate XA is the frozen exact/skeleton/7-neighbour hierarchy with its public-validation-only
gates. Candidate XB is the fixed public-only LinearSVC. Candidate XC is three-seed BanglaBERT with
public-validation checkpoint selection, fixed threshold 0.50, and no official fitting. Candidate XD
uses only eligible XA transfers and otherwise the authenticated V4-A prediction. At most the named
XD primary and XC secondary submissions can be generated, and only under the predeclared gates.

## Execution results

The three public files authenticated exactly. The contrastive file was included as a legitimate
balanced 5,000-row resource, but it contains all 1,000 validation pairs and extensive same-label
duplication with train. Validation matches, 36 mixed-label pair rows, and 3,982 same-label duplicate
rows were removed. The final training pool contains 3,978 rows (label 0/1: 1,991/1,987), manifest
`f16a2658e8e34b07ce77bfc58cb7a00be48f00e38196a2529d122d45866dcf84`.
There are zero exact pair, prompt, response, triple, or skeleton overlaps between that pool and the
official sample, so the conservative official-safe pool remains 3,978 rows.

Public-validation macro F1 was XA 0.480675, XB 0.426793, and XC 0.493339. No exact, skeleton,
or nearest-neighbour transfer rule passed its public-validation support/precision gate. Official
external-diagnostic macro F1 was XA 0.535034, XB 0.438431, XC 0.499335, and XD 0.687270. XD made
zero transfers and therefore exactly reproduced authenticated V4-A OOF predictions.

Public-to-test audit found zero exact prompt, response, pair, triple, or skeleton matches. Only one
of 2,516 test rows reached similarity 0.70, none reached 0.75, and no transfer route was eligible.
XD Gate A, XD Gate B, and the XC submission gate all failed. No submission CSV was generated,
and no further V13 experiment was run.

The initial run trained all three XC seeds and selected their public-validation checkpoints, then
hit a narrow finalization bug because checkpoint creation had already created the experiment root.
The checkpoints were preserved and authenticated; a resume recomputed validation/official
probabilities without retraining. Initial runtime was 1,603.3 seconds and resume/finalization was
56.45 seconds. The selected synthetic batch size was 64. Peak allocated VRAM was 11,498,987,008
bytes, below 90% of the 12,878,086,144-byte GPU. Final process peak working set was
2,538,258,432 bytes. Six normalization/similarity workers and four DataLoader workers were used.
