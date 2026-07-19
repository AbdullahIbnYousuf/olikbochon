# V7 Error Analysis

This analysis uses only official labeled training OOF predictions. No competition test examples
were loaded, displayed, or summarized. Categories below are analysis targets, not manually
asserted ground-truth error causes.

| Error category | Observable diagnostic | V7 finding / limitation |
|---|---|---|
| Entity errors | Entity-token support/overlap | Compact lexical signals exist; no entity-level gold annotations |
| Number errors | Response/context number support | E0/E1 numeric features cover this; category-specific F1 is not identifiable from labels |
| Date errors | Date-like numeric support | Included in numeric channel; no separate date labels |
| Location errors | Location-token support | No authenticated Bengali NER resource was added |
| Relation errors | Prompt/evidence/response lexical relations | Frozen semantic arm failed to improve this indirectly |
| Negation errors | Negation-marker mismatch | Available for future aggregate diagnostics; no causal annotation |
| Unsupported additions | Low evidence overlap / response expansion | E0 lexical and E2 evidence-conditioned inputs cover the signal |
| Mixed faithful/hallucinated | Full-response label only | Atomic claims are not implemented or claimed |
| Retrieval failure | Rejected retrieval / low score | Preserved per E2 OOF row and fold threshold |
| Wrong evidence retrieval | Accepted but irrelevant evidence | No passage-relevance labels; cannot measure directly |
| Evidence available, verifier failure | E2 wrong with accepted/official evidence | E2 overall error count was 424/897 |
| Context sentinel/routing failure | Central `has_context` test matrix | All observed/supported sentinels pass; 130 present and 169 absent rows |

## OOF error overlap

| Pair | Both wrong | Left-only wrong | Right-only wrong | Disagreements |
|---|---:|---:|---:|---:|
| E0 / E1 | 146 | 133 | 118 | 251 |
| E0 / E2 | 152 | 127 | 272 | 399 |
| E0 / E3 | 210 | 69 | 58 | 127 |
| E0 / E4 | 216 | 63 | 69 | 132 |
| E1 / E2 | 148 | 116 | 276 | 392 |
| E1 / E4 | 182 | 82 | 103 | 185 |
| E3 / E4 | 222 | 46 | 63 | 109 |

E2 corrected 127 E0 errors, so it is not identical signal; however, it introduced 272 errors on
rows E0 got right. Nested E4 could not turn that noisy complementarity into a stable gain.
Aggregate error counts were E0 279, E1 264, E2 424, E3 268, and E4 285 out of 897 repeated OOF
decisions.
