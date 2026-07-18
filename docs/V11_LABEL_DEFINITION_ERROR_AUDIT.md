# V11 label-definition and error-family audit

## Scope and safety

V11 is a discovery-only audit of the authenticated 299-row official sample. It trained or
evaluated no new candidate, accessed no competition-test row, created no test prediction or
submission, and used no leaderboard result to define a rule. The corrected route totals were
130 context-present rows (labels 0/1: 47/83) and 169 context-absent rows (labels 0/1: 89/80).
The local row-level table is ignored under `artifacts/v11/`; this tracked report contains no raw
prompt, response, or context text.

## Locked audit split

The strengthened V5 family/group graph was used before content taxonomy or model outcomes were
inspected. The first feasible seed in the frozen sequence was `20260719`.

| Partition | Rows | Groups | Label 0 | Label 1 | Row-index SHA-256 |
|---|---:|---:|---:|---:|---|
| Discovery | 127 | 125 | 68 | 59 | `775d9a0170fa4e5aacceb1815243ed58cb39beced475b0269ffb911872146320` |
| Locked holdout | 42 | 42 | 21 | 21 | `c0ce983882e73778dc0de857c8ed708c518111318c03ef998d739289124e42db` |

Group overlap was zero. The full per-group hash manifest is retained locally in `results.json`.
No holdout prompt/response text was inspected or displayed, no holdout I/R/U prediction was
generated, and no holdout metric was computed. The blind discovery taxonomy hash is
`e87fafeadbde3c1b903b4388cf72bb6a2f07f6597b591c0685376e5d98d0b20a`.

## Blind taxonomy and structural findings

Taxonomy was completed before labels or OOF outcomes were joined. Deterministic operational
rules classified 77 rows as factual entity questions, 31 as numeric/date/quantity/ranking, 7
as creative/opinion/open-ended, 7 as other, and 5 across the remaining represented families.
The subjective boundaries were frozen as follows: an ambiguous prompt is shorter than 12
normalized characters or matches the generic-query lexicon; a generic low-information response
is shorter than 25 normalized characters; unverifiable specificity requires more than 160
normalized characters plus a named-entity-like or numeric claim.

Discovery responses were unusually compact: mean character length 14.0394, token estimate
2.3701, number count 0.3543, sentence count 1.0236, prompt-response lexical overlap 0.0113, and
repetition ratio 0.0046. No URL/citation, refusal, or hedging flag fired. The most frequent
support-qualified risk tags were generic low-information (108), named entity (86), numeric claim
(32), malformed/incomplete (20), geographic (19), financial (11), date/temporal (8), and
superlative/ranking (6). These deterministic tags are audit descriptors, not corrected labels.

## Genuine OOF provenance and discovery metrics

Candidate I and Candidate R were regenerated under their unchanged seeds 17/29/43 and five
strengthened group folds. Every discovery row was predicted exactly once per seed by a model
that excluded its strengthened group; the three genuine OOF probabilities were then averaged
per row. Candidate U is the frozen 50/50 I/R probability average. All probabilities are oriented
as P(label 1), the decision threshold is fixed at 0.50, alignment is one-to-one, and there are no
missing or duplicate indices. Holdout predictions generated: zero.

| Candidate | Macro F1 | Label-0 F1 | Label-1 F1 | Accuracy | Confusion matrix | Predicted 0/1 | Brier |
|---|---:|---:|---:|---:|---|---:|---:|
| I | 0.558371 | 0.540984 | 0.575758 | 0.559055 | `[[33,35],[21,38]]` | 54/73 | 0.271680 |
| R | 0.551181 | 0.551181 | 0.551181 | 0.551181 | `[[35,33],[24,35]]` | 59/68 | 0.253197 |
| U | 0.527530 | 0.531250 | 0.523810 | 0.527559 | `[[34,34],[26,33]]` | 60/67 | 0.249939 |

The highest-impact families by support times Candidate-U error rate were factual entity questions
(77 support, 32 errors), numeric/date/quantity/ranking (31, 18), creative/opinion/open-ended
(7, 5), and other (7, 2). Small one-row families were not promoted above these larger groups.
Within the 32 numeric-claim rows, U had a 0.4444 false-positive rate and 0.7143 false-negative
rate. Within the 86 named-entity rows, the corresponding rates were 0.5111 and 0.3659.

## Disagreement and duplicate findings

I and R disagreed on 40.1575% of discovery rows. Counts were: I-only correct 26, R-only correct
25, both correct 45, and both wrong 31. Probability correlation was only 0.185672. The frozen
review condition selected 82 rows; every selected row received exactly one deterministic error
reason. Counts were generic-but-acceptable 47, numeric/date inconsistency 22, lexical shortcut
5, semantic shortcut 4, incorrect entity/relation 2, ambiguous annotation 1, and suspected label
inconsistency 1. Thus the aggregate suspected-ambiguity count is 2. These descriptions are
hypotheses: they do not establish an annotation error.

Discovery-only duplicate analysis found two exact-prompt clusters (4 rows, both mixed-label),
two exact-response clusters (4 rows, both mixed-label), no exact or normalized prompt-response
duplicate cluster, and one near-duplicate cluster at character TF-IDF cosine >= 0.90 (2 rows,
mixed-label). I/R/U error rates inside any duplicate or near-duplicate cluster were
0.5000/0.3750/0.3750, versus 0.4370/0.4538/0.4790 outside.

## Operational label interpretation

The discovery evidence does not support treating the labels as a simple “factually correct
versus incorrect” distinction. Operationally, label 0 marks hallucinated or otherwise
unacceptable responses and can include factual error, unsupported or unverifiable specificity,
irrelevance, or instruction noncompliance. Label 1 marks an acceptable or faithful response and
can include generic, uncertain, creative, or subjective answers when those fit the request.
Refusal and uncertainty must be separated from unsupported certainty; creative responses must be
judged against the instruction rather than external factuality; a generic answer may be
non-hallucinatory yet unhelpful. Mixed-label duplicate clusters and the two ambiguity-tagged
review cases are potentially inconsistent with a pure factuality definition and require organizer
clarification. V11 does not claim that either label is wrong.

## Frozen V12 hypotheses (not evaluated)

Only discovery-supported hypotheses are frozen, and V11 reports no effect estimate for them.

1. **Numeric/date consistency** — target the 32 numeric-claim rows by counting normalized
   number/date mentions and flagging response-only values. Expected direction: fewer numeric
   false decisions. Candidate I lacks explicit consistency and R compresses it into semantic
   aggregates. Leakage risk is low with train-partition fitting; overfitting risk is moderate;
   no external evidence is required; it is automatic on test rows; minimum support is 5.
2. **Generic-answer detection** — target the 108 short-response rows with the exact rule
   “normalized response shorter than 25 characters and low prompt lexical overlap.” Expected
   direction: distinguish acceptable generic answers from mismatches. Leakage is low, overfitting
   moderate, no external evidence is required, test computation is automatic, minimum support 5.
3. **Factual-specificity risk** — target the 86 named-entity rows using deterministic counts of
   named-entity-like and numeric response tokens. Expected direction: reduce unsupported
   specificity errors. External evidence may be required for final adjudication; leakage is low,
   overfitting moderate, test computation automatic, minimum support 5.
4. **Prompt-task-family routing** — target the 77 factual-entity rows with the already frozen
   twelve-family prompt classifier. Expected direction: a predeclared family-specific correction;
   both base models lack an explicit family decision. Leakage is low if rules stay frozen,
   overfitting risk is high, no external evidence is intrinsically required, test computation is
   automatic, minimum support 5. It may receive one confirmation on the locked V12 holdout only.

No hypothesis, threshold, routing correction, or new candidate was evaluated in V11.

## Resources

The audit completed in 130.971 seconds with four deterministic tagging workers, four similarity
workers, no nested process pools, BLAS thread counts fixed at one, and one offline mDeBERTa GPU
process at batch size 64 for frozen Candidate-R feature regeneration. Peak process working set was
3,513,339,904 bytes; the final measured system memory fraction was 0.47.
