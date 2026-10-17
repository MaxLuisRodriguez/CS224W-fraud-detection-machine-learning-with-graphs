# Final research question — September 24, 2026

> Can reasoning about shared accounts reduce avoidable evidence-search failures as transaction graphs grow, beyond strong simple and nonmyopic policies at the same query budget?

An investigator can retrieve only a few account histories. As more transactions create more possible trails, connections among the records already retrieved may help it choose where to look next. The question is whether it uses those connections well enough to find evidence its observations actually make findable.

A successful query path existing somewhere in the graph is insufficient. With eight indistinguishable branches and one hidden witness, four queries can succeed only half the time, even though each realized graph has a successful one-query path. An investigator should not be judged against that clairvoyant path.

The project therefore begins with tiny graph distributions where we can compute the best policy using the same query responses. We measure how much performance declines because the information problem became harder, and how much additional decline comes from the tested policy. Shared-account relationships are the mechanism to test, not an assumed reason a GNN must win.

## Chosen direction

Analysis of relational evidence acquisition under ordinary graph growth. The initial deliverable is a trustworthy measurement of avoidable search failure. A graph policy follows if strong simple and planning baselines leave meaningful room for improvement.

The first witness is two internally disjoint, three-edge temporal paths from the target sender to the same downstream account. All supporting records must be retrieved. This is a graph-property witness, not proof of laundering. Five queries can reveal it; the main budget is six calls.

Compare exact-reference success with random, breadth-first, recency, path-completion/diversity, and nonmyopic planning. If justified, compare a PyG policy with an identity-aware set/attention model given the same observations and training supervision. Test clean-only and growth-augmented training for both.

## Why it could matter

The Socratix take-home enforced what an agent may retrieve. This research asks whether permitted retrieval choices uncover useful evidence. The potential product contribution is an evaluation harness and, if supported, a better query selector. The per-account retrieval bottleneck remains a product assumption.

The publication bet is a general account of when graph growth creates avoidable acquisition failure and what fixes it. The exact-reference equations, a GNN, or a successful toy demonstration alone do not establish novelty. Existing graph exploration and active acquisition work substantially constrain the claim.

## What would disprove the preferred story

- The same-observation optimum deteriorates enough to explain the observed loss.
- A simple path-completion/diversity policy already achieves the useful ceiling.
- Improvement disappears when competitors receive the same identity information, training data, and tuning effort.
- The effect fails on independent structures or relies on generator shortcuts.

The [active plan](PROJECT_PLAN.md) contains the contract, references, decision gates, 44-hour pilot, and later public-data stage. There are no trained models or experimental results. The first programming milestone is the gateway, symbolic witness verifier, and exact observation-constrained planner after writing the finite completion tables.
