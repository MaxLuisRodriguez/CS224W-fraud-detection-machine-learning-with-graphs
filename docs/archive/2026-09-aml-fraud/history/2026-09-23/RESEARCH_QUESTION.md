# What should a financial-crime agent inspect next?

> Status: superseded candidate, retained for context. The active specification is
> [PROJECT_PLAN.md](PROJECT_PLAN.md); [RESEARCH_DECISION.md](RESEARCH_DECISION.md)
> records the reason for the change. The original method remains a baseline.

## The problem in plain language

A financial-crime agent receives a suspicious transfer. It can request account
histories, but each request takes time, returns more information to process, and
may reveal further accounts worth investigating. Which history should it request
next to uncover useful evidence with fewer requests?

For example, separately ordinary branches of transfers might meet at the same
intermediary. Recognizing that it is the **same account** can change which history
is worth requesting next. This is evidence of a pattern, not proof of criminality:
the experiments must include legitimate lookalikes.

## The research question

**Does explicitly preserving shared accounts and connections across investigation
branches improve sequential evidence acquisition, compared with methods that
organize evidence as separate lists or trees?**

The primary outcome is held-out detection performance versus the number of
history requests. The environment fixes the historical cutoff, page size, available
sources, and initial information for every method. It measures logical requests
and actual runtime separately; simulated request cost is not a claim about a bank's
API pricing or analyst time.

## Why Socratix might care

An agent that investigates financial crime must decide what to inspect, as well as
interpret what it has inspected. A successful acquisition component could reduce
redundant retrieval and focus an agent's evidence gathering. This is a product
hypothesis based on Socratix's public description, not knowledge of its internal
architecture or an established customer requirement.

## What is and is not novel

Graph search, changing action sets, paid queries, missing evidence, nonmyopic
acquisition, and abstention already have substantial precedents. Combining them
in a fraud demo does not establish a new research contribution.

The candidate contribution is a **relational acquisition policy** that uses shared
entities, cross-branch connections, and query-completeness state to estimate the
downstream value of evidence requests. Its novelty remains unproven. In particular,
adding status embeddings to a GNN is not sufficient.

The interview implementation first supplies a transparent one-step policy and
strong simple baselines. Longer-horizon acquisition is a separate experiment. We
must identify an advantage over structured acquisition and graph-search methods,
not just over random queries.

## Falsifiable claims

1. At the same query budget, relational acquisition improves held-out average
   precision over strong fixed-order and structured acquisition baselines.
2. Removing cross-branch identity information reduces that advantage on controlled
   examples requiring relational evidence. This is an information ablation, not
   an equal-information method comparison.
3. Longer planning horizons help when the first query is useful primarily because
   it unlocks a later request.
4. Benefits persist under unseen source-availability masks and beyond one synthetic
   financial dataset.

If these claims fail, publish the result honestly as an empirical study and revise
the research claim. A working demo is not evidence of main-conference readiness.

## Closest work to beat or distinguish

- [Hierarchical costly-feature acquisition (2024)](https://link.springer.com/article/10.1007/s10994-024-06565-4): structured data, dynamic actions, acquisition and stopping.
- [ACO (ICML 2024)](https://proceedings.mlr.press/v235/valancius24a.html): nongreedy feature acquisition.
- [AgentGL (2026 preprint)](https://arxiv.org/abs/2604.05846): learned graph-tool exploration.
- [KnowGuard (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8443219a991f068c34d9491ad68ffa94-Abstract-Conference.html): acquiring graph evidence before abstaining.
- [CDEG (2026 preprint)](https://arxiv.org/abs/2608.22899): evidence-action relations for diagnostic agents.

Research positioning last reviewed: 2026-09-22. Recheck before submission.
