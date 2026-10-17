# Related-work audit and contribution boundary

Audit date: September 25, 2026. This document supersedes the earlier claim that common-continuation value learning might itself distinguish the method. This is a targeted primary-literature audit, not an exhaustive novelty clearance or a reproduction of the cited experiments.

## Decision

Keep the application and shared-evidence protocol. **Do not present the existing dense value-head model as a novel algorithm.** The strongest remaining candidate is efficient joint acquisition when the set of cases helped by a query is initially uncertain and changes as the graph is revealed. Whether a useful new method exists here remains unresolved.

The publication question is:

> Can we learn which unresolved cases an evidence request could help, including through later requests, and use that information to match dense joint planning with substantially less computation as the queue grows?

The financial objective remains detection under a retrieval budget. Computational savings alone are insufficient if the dense baseline was unnecessarily expensive: a scalar global value model is a mandatory competitor. An exact or empirical result about this problem could support a paper; merely combining existing components cannot establish one.

## Closest work and consequences

Each linked source is a paper, author manuscript, or official proceedings record. Preprints are identified as such; recent versions do not establish conference acceptance.

| Work and inspection scope | Already established | Consequence for this project |
|---|---|---|
| [Effective Label Acquisition for Collective Classification, KDD 2008](https://linqs.org/assets/resources/bilgic-kdd08.pdf); [Active Inference for Collective Classification, AAAI 2010](https://ojs.aaai.org/index.php/AAAI/article/view/7704). Primary paper / official abstract. | Acquisitions can improve predictions for related entities. | Cross-case benefit is old. Here queries release transaction records, not labels; that changes the interface, not automatically the scientific contribution. |
| [Optimal Nonmyopic Value of Information, IJCAI 2005](https://www.ijcai.org/Proceedings/05/Papers/1154.pdf). Primary formulation. | Sequential information gathering in graphical models. | Delayed utility and conditional observation plans are inherited ideas. |
| [CAMA, arXiv v4, May 2026](https://arxiv.org/html/2505.16791v4). Section 3, equations 1–4 and acquisition strategy table. | Acquisition allocation across a cohort using a cohort performance objective. | A global alert budget is not new. Its stated score update changes selected samples' scores; our request can change multiple scores and reveal new requests. Adapt its acquisition rules to allow that, rather than preserving a disadvantage. |
| [BRiG-AFA, August 2026 preprint](https://arxiv.org/html/2608.02305). Sections 3–4 and limitations. | Frozen predictor, budget-specific Bellman risk regression, offline targets and nonmyopic acquisition. | This directly overlaps the proposed training recipe. Build a scalar queue-risk adaptation first. Do not claim fitted Bellman acquisition as our contribution. Its published experiments do not establish superiority over every modern AFA method. |
| [Non-Myopic Active Feature Acquisition via Pathwise Policy Gradients, May 2026 preprint](https://arxiv.org/html/2605.05511). Sections 2–3. | Finite-horizon acquisition and stopping trained through relaxed actions and masks. | A recent alternative to value fitting. Test on the controlled fixed-candidate setting; document the extra work needed to support newly discovered graph actions. A masked PPO comparator alone does not cover this method. |
| [PEGE, arXiv v2, June 2026](https://arxiv.org/html/2601.16233v2). SAPOG definition and approach. | Sequential decisions on a partially revealed graph, with state-dependent reward and sampled expansions. | Partially observed topology is established. Our interface differs from its induced-subgraph and label-revealing observation model. Give its planner our observation rules and joint reward. Generic reward functions can express multiple-case utility. |
| [ACO, ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html); [SEFA, ICML 2025](https://proceedings.mlr.press/v267/norcliffe25a.html). Proceedings summaries; full adaptation still pending. | Nongreedy acquisition alternatives. | Include a faithfully reproduced nonmyopic acquisition family; report which aspects require adaptation. Neither paper is dismissed because it uses tabular benchmarks. |
| [Costly features in hierarchical deep sets, 2024](https://link.springer.com/article/10.1007/s10994-024-06565-4). Earlier formulation review. | Structured sequential acquisition with newly accessible actions. | Hierarchy, API calls and changing frontiers are not novelty. Include an identity-aware structured baseline with identical metadata. |
| [Successor Features for Transfer in RL, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/350db081a661525235354dd3e19b8c05-Paper.pdf). Linear reward decomposition. | Component returns under a common policy can be combined to evaluate a total reward. | Our per-alert losses are reward components. Their mean and shared next-action target are inherited machinery, not a new solution to incompatible continuations. Transfer guarantees do not automatically extend to our learned partial-graph state. |
| [Hybrid Reward Architecture, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/1264a061d82a2edae1574b07249800d6-Abstract.html), with [official reviews](https://proceedings.neurips.cc/paper/2017/file/1264a061d82a2edae1574b07249800d6-Reviews.html). Reward decomposition and historical discussion. | Multiple value heads and aggregation were explicitly studied. | Cite as precedent. Review discussion is not evidence that a suggested variation was implemented or experimentally validated. |
| [Deep Coordination Graphs, ICML 2020](https://proceedings.mlr.press/v119/boehmer20a.html). Official method summary. | Factored values, graph message passing, parameter sharing and efficient coordinated decisions. | Sparse graph value models are not new generally. Our single evidence action is different from a multi-agent joint action, but that distinction alone does not establish a new algorithm. |
| [Zero-Shot AFA via LLM-Elicitation, June 2026 preprint, v2](https://arxiv.org/html/2606.18933). Sections 1–3 and appendix headings. | Shared-budget top-k decisions; feature support can affect multiple pairwise comparisons; pruning is discussed. | Neither multiple beneficiaries nor selective computation is automatically new. Before claiming adaptive influence pruning, inspect and compare its appendix D/E pruning rules. This audit has not reproduced those rules or established that our candidate escapes them. |

## Recent work that should inform, but not define, the claim

[AgentGL](https://arxiv.org/abs/2604.05846v2) overlaps tool-mediated graph exploration. [DynaKRAG, July 2026 preprint](https://arxiv.org/abs/2607.06507) overlaps learned evidence control for multi-hop retrieval. Their application differs, so they are context rather than substitutes for the acquisition baselines above. An agent interface or an LLM narrator is not the contribution.

The [IJCAI 2026 AFA survey](https://www.ijcai.org/proceedings/2026/860) is a useful citation map. Search also surfaced [explanation-driven AFA for recourse, September 2026](https://arxiv.org/abs/2609.12179); only its abstract was screened here. Recourse does not appear to be the central comparator. Recency should not displace an older paper with a closer formulation.

## Specific candidate to investigate, not yet a cleared novelty claim

Develop adaptive computation over action–case influence, on top of standard joint value learning:

1. A dense reference estimates each candidate request's future effect on every alert under one queue policy.
2. A cheaper router proposes potentially affected alerts from released identity links, embeddings and budget. No unseen edges or labels enter routing.
3. A residual model estimates the total contribution of omitted alerts. Candidate rankings are refined by evaluating more alert contributions when omission could plausibly change the chosen request; dense fallback is permitted.
4. After each real response, update the routing state, including newly connected cases. Do not permanently discard an alert just because it is currently disconnected.
5. Train and validate omission estimates on disjoint training/validation trajectories. In deployment, routing uses observations only. Learned intervals are empirical diagnostics, not certified bounds.

The mechanism is preserving *delayed* cross-case value while reducing unnecessary action–case evaluation. A one-hop neighborhood mask or generic top-k attention is an ablation, not a new method. Important failure regimes are dense overlap, newly discovered bridges, uncertain distant benefits, and small queues where routing overhead exceeds its savings.

Do not assume per-case effects are nonnegative: information can worsen a fixed predictor's realized loss. Omission tests must account for signed contributions. A sparse model may save value-head computation while predictor recomputation still dominates total runtime; measure the complete pipeline.

For analytical work, under an explicitly assumed uniform action-value error epsilon, an approximate greedy choice loses at most 2 epsilon against the exact greedy choice. That elementary fact is not a novel theorem and is not a certificate for learned estimates. The substantive question is whether useful observable bounds or empirically reliable refinements can be obtained without doing the dense work. Drop theoretical claims if that cannot be established.

## Experiments required before calling this a contribution

- Adapt scalar BRiG-style terminal-risk regression and dense common-policy reward decomposition to the identical queue protocol. Compare with a scalar GNN and scalar attention policy; neither requires an action-by-case output matrix.
- Compare adaptive refinement against fixed top-k routing, random routing, observed-neighborhood masks, low-rank factorization, and dense scoring. Give all methods global evidence, shared cache and the same tuning resources.
- Use tiny worlds to vary overlap and delayed benefit separately, including no-overlap, irrelevant shared hubs and newly discovered bridges. Compare with a nonclairvoyant exact policy under the same prior.
- Measure retrieval calls, retrieved rows, prediction metrics, whole-pipeline latency and total training cost separately. Plot both detection–retrieval and detection–runtime frontiers.
- Predeclare a provisional engineering target: at least 2x lower end-to-end acquisition-decision latency with at most one absolute percentage point loss in recall at the fixed review capacity, at matched retrieval cost, relative to the strongest dense comparator. Separately require a benefit over efficient scalar and heuristic competitors. These targets are not statistical significance thresholds or forecasts.
- Require natural overlap and useful historical signal before scaling. Confirm on an independent financial generator and held-out queue sizes. Report dense-regime failures rather than tuning them away.

## What is and is not resolved

Resolved: the previous dense algorithm is not a defensible standalone novelty claim; the plan now has closer, newer mandatory comparators and a concrete candidate mechanism.

Unresolved: whether adaptive influence refinement is itself sufficiently distinct, whether its assumptions hold, and whether it improves a strong scalar joint model. The zero-shot AFA pruning appendix, adaptive coordination-graph literature, and computation-aware planning are explicit remaining novelty checks before major spending. A search failing to find an exact title match is not proof of originality.

Decision: proceed with the bounded feasibility/baseline phase, not a claim of conference readiness. If ordinary joint acquisition works well, the honest result is a useful application or benchmark study. Do not keep adding terminology to turn an already-solved method into a purported invention.
