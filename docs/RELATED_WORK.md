# Related work and contribution boundary

Targeted primary-source check, 2026-10-10. Every item was opened (paper, official abstract, or proceedings page). Preprints are marked. This is not an exhaustive novelty clearance; repeat the search before submitting anywhere.

> **Name collision:** "Scout" (Zhou, arXiv 2609.05862) is an unrelated academic method, not the SCOUT pre-call brief agent. In writing, cite it as *Zhou (2026)*.

## Closest work

| Work | Status | Overlap | What stays distinct here |
|---|---|---|---|
| [Scout: Budgeted Task-Aware Acquisition of Dynamic Networks](https://arxiv.org/abs/2609.05862) (Zhou) | Preprint, Sep 2026 | A GNN learns which nodes to query under a budget; utility = singleton counterfactual change in task score; beats random/degree/uncertainty | Homogeneous dynamic graphs, refreshing *stale* state; scores each query independently (no interaction between lookups); no designated target batch; no relational/heterogeneous setting; no oracle bound. **Must be reimplemented as a baseline.** |
| [CAMA](https://arxiv.org/abs/2505.16791) (Rheude, Eils, Wild) | Preprint | Cohort-level test-time acquisition under a budget; oracle-greedy upper bound | Each acquisition helps only its own sample; no graph or shared entities |
| [BRiG-AFA](https://arxiv.org/abs/2608.02305) | Preprint, Aug 2026 | Nonmyopic acquisition via Bellman risk-to-go | Per instance, flat features |
| [ACO](https://proceedings.mlr.press/v235/valancius24a.html), ICML 2024; [SEFA](https://proceedings.mlr.press/v267/norcliffe25a.html), ICML 2025 | Peer-reviewed | Strong per-instance active feature acquisition | Per instance, no graph; use one as the per-target baseline |
| [Active Inference for Collective Classification](https://ojs.aaai.org/index.php/AAAI/article/view/7704) (Bilgic & Getoor), AAAI 2010 | Peer-reviewed | One test-time acquisition improves neighboring predictions | Acquires *labels*, not rows; pre-GNN |
| [KumoRFM-2](https://arxiv.org/abs/2604.12596) | Preprint, Apr 2026 | Robustness to random value masking and up to 75% foreign-key removal on relational data | Removal is random, not chosen or budgeted |
| [STEER](https://arxiv.org/abs/2610.00907) | Preprint, Oct 2026 | ~40% less inference context at equal accuracy on RelBench, via LLM-ranked schema edges | Compute-motivated, schema-level, full database accessible |
| [LightRDL](https://arxiv.org/abs/2504.04934) | Preprint, 2025 | Reduced relational graph for faster inference | Efficiency only |
| [RDB2G-Bench](https://arxiv.org/abs/2506.01360), NeurIPS 2025 D&B | Peer-reviewed | Which tables/relations to include in the graph | Schema-level, training time |
| [Feature Propagation](https://proceedings.mlr.press/v198/rossi22a.html), LoG 2022; [GNNmim](https://arxiv.org/abs/2601.04855) (preprint) | Mixed | GNNs with missing node features; mask-style training | No acquisition; don't claim mask training as new |
| [NAC / selective network discovery](https://arxiv.org/abs/1909.07294) | Preprint | RL chooses node queries that reveal links | Goal is node harvesting, not prediction quality |
| [BATS / Budget Tracker](https://arxiv.org/abs/2511.17006), COLM 2026 | Peer-reviewed | Tool-call budgets for search agents; cost–quality Pareto curves | Text QA, no graph |
| [RelAgent](https://arxiv.org/abs/2605.07840) | Preprint, May 2026 | LLM agent on RelBench | Budget limits *model search*, not inference-time lookups |

Infrastructure and backbones to cite:

- [RelBench](https://arxiv.org/abs/2407.20060), NeurIPS 2024 D&B
- [RelBench v2](https://arxiv.org/abs/2602.12606)
- [RelGNN](https://proceedings.mlr.press/v267/chen25ad.html), ICML 2025
- [RelGT](https://arxiv.org/abs/2505.10960)
- [Griffin](https://arxiv.org/abs/2505.05568)
- [Relational Transformer](https://arxiv.org/abs/2510.06377)
- [Rel-LLM](https://aclanthology.org/2025.acl-long.386), ACL 2025

B2B lead-scoring papers found (gradient boosting, CatBoost, one spectral GCN) do not study acquisition.

## Claim boundaries (use this wording)

- **C1 — retrieval-budgeted RDL.** No budgeted, per-row, policy-chosen retrieval frontier for RDL was found. Random-removal robustness (KumoRFM-2) and compute-motivated context reduction (STEER, LightRDL) exist and must be cited.
  - Claim the *protocol and empirical finding*, not "first to study partial information".
  - Mask training itself is known.
- **C2 — batch-aware acquisition with shared-entity credit.** This is **partially done** by Zhou (2026), which already has a learned GNN query scorer with task-level counterfactual credit.
  - The defensible difference is **interaction-aware value on heterogeneous relational graphs with a designated target batch**:
    - a `get` is worth more after `list`s that reveal who references the entity;
    - shared stubs couple many targets;
    - Zhou (2026) explicitly scores queries one at a time.
  - Required experiment: our learned value vs. a faithful Zhou-style singleton scorer, on the same predictor and cache.
- **C3 — structure discovery + enrichment, nonmyopic, heterogeneous costs.** The pieces exist separately (NAC, Zhou 2026, BRiG-AFA/ACO). No combination for prediction quality on relational graphs was found.

## Venue facts (official sites, checked 2026-10-10)

- **LoG 2026:** paper deadlines (Aug 2026) and journal track (Sep 30) have passed.
- **KDD 2027:** first cycle passed (Jul 2026); second-cycle dates not yet posted.
- **ICML 2027, NeurIPS 2027:** not yet posted. Dates on aggregator sites are projections.
