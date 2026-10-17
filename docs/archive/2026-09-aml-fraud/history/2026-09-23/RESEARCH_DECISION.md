# Research direction review — 2026-09-22

The user has requested a restructured plan. [PROJECT_PLAN.md](PROJECT_PLAN.md)
is now the active specification. This file retains the rationale for the pivot.

## Current decision

Replace expansion of the original acquisition model with the paired failure study.
The user has made main-track
NeurIPS/ICLR/ICML publication the overriding research objective and questioned
whether the initial contribution is sufficiently novel. The data and evidence
environment are reusable infrastructure; they are not evidence of novelty or a
completed experiment. No research results have been produced.

## Stronger candidate to test first

**Can a graph investigator be diverted from available, diagnostic evidence by
strategically placed distractors, and can we learn investigation policies that
resist this without losing their efficiency on ordinary cases?**

The distinguishing experiment must preserve the original target, its label under
the declared synthetic perturbation model, and the original supporting evidence.
It must establish that diagnostic evidence remains reachable under the same API
and query budget. It must isolate choosing the wrong queries from misclassifying
the evidence returned by correct queries.

This is a candidate, not a proven first contribution. Adversarial active learning,
fraud camouflage, graph poisoning, and attacks on GraphRAG all precede it.

## First research gate (approximately 3 focused days)

1. Build a small controlled graph family with an explicit diagnostic witness,
   benign lookalikes, and distractors. Keep full-information labels unchanged.
2. Compare a clean graph, matched random distractors, and policy-targeted
   distractors. Use identical page sizes, source scopes, and query budgets.
3. Separate two stages experimentally: acquisition and final prediction. Replay
   a valid clean query sequence on the modified graph. Count a policy-specific
   failure only where that sequence remains legal, fits the same budget, and
   retrieves diagnostic evidence. Report all cases as well as this subset.
4. Test breadth-first, random, diversity-based, and learned policies. A learned
   policy losing only to random on a contrived graph is not enough.
5. Stop treating this as the primary novelty claim if the effect is only page
   truncation, unlimited decoy volume, classifier poisoning, or easily eliminated
   by a simple matched-budget baseline.

The main-paper path requires a substantive robust acquisition method or analysis,
multiple independently meaningful graph settings, and a fresh closest-work review.
The interview can present the verified failure and controlled measurements before
any larger contribution is established. Do not claim realistic financial attacks
from unconstrained synthetic graph edits.

## Primary references that constrain the claim

- [Relational active feature acquisition, 2023](https://www.dfki.de/en/web/research/projects-and-publications/publication/14022)
- [Adversarial active learning, 2014](https://istc-cc.cmu.edu/publications/papers/2014/aisec14-active.pdf)
- [AgentGL, 2026 preprint](https://arxiv.org/abs/2604.05846)
- [TransXion, 2026 preprint](https://arxiv.org/abs/2604.17420)
- [LogicPoison, ACL 2026](https://aclanthology.org/2026.acl-long.252/)
- [KEPo, 2026 preprint](https://arxiv.org/abs/2603.11501)
- [Deception Game, CoRL 2023](https://proceedings.mlr.press/v229/hu23b/hu23b.pdf)

## Alternatives considered

- Shared evidence across multiple simultaneous cases: appealing product fit, but
  distinguish it carefully from batch acquisition, caching, and set coverage.
- Counterfactual evaluation under selective labels: important, but substantial
  identification assumptions and existing AFA/OPE literature make it a riskier
  solo project without real investigation logs.
- Evidence double-counting in agent memory: useful, but provenance and deduplication
  alone do not establish a general graph-learning contribution.
