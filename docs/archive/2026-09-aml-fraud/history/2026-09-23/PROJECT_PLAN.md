# Can Graph Investigators Be Distracted?

**Robust evidence acquisition under a fixed investigation budget**

Status: active project plan, replacing the original acquisition-efficiency plan.
Updated: 2026-09-22. Research novelty and performance are not yet established.

## 1. The problem, motivation, and intended contribution

### Plain-language motivation

A financial-crime investigator starts with a suspicious transfer and follows
related account histories. It has limited time to investigate. Additional branches
can make an unhelpful trail look promising. The investigator may then miss a
diagnostic relationship even though the underlying evidence is still available.

**Question: Can distracting graph structure cause an investigator to miss evidence
that remains reachable within its query budget, and can a learned policy resist
that distraction without sacrificing ordinary-case efficiency?**

An illustrative example uses eight history requests: an ordinary investigation
finds relevant evidence in six requests; on a modified graph, another valid
six-request sequence still finds that evidence, but the agent spends all eight
requests elsewhere. This is an illustration, not a measured result.

### Why this matters for Socratix

Socratix publicly describes agents that gather evidence and investigate financial
crime. This project tests the reliability of that evidence-gathering step. A
successful result could contribute a stress-test harness and a robust query
selector for an investigation tool layer. It does not assume access to Socratix
data or identify an actual weakness in its product.

The bridge from the completed take-home is precise: the take-home enforced what
an agent may retrieve; this research tests whether its permitted retrieval choices
lead it to useful evidence. Reuse the design principles of scoped tools, validated
releases, and replayable traces. Do not reuse company fixtures as research data.

### What a paper would contribute

1. **A controlled evaluation of search failures.** Paired investigations distinguish
   retrieval-path failures from missing evidence, page overflow, and errors in the
   final classifier.
2. **A robust acquisition method.** Learn query selection that maintains useful
   evidence discovery under bounded distracting additions, evaluated against strong
   acquisition and simple exploration baselines.
3. **Evidence of generality.** Verify the mechanism on controlled graph tasks,
   public transaction data, and a second graph domain.

A new benchmark and an attack demonstration alone do not establish the desired
main-track contribution. Standard adversarial training alone may also be
insufficient. The final claim must explain a general mechanism or supply a
substantive methodological or analytical result supported by experiments.

### Closest work and novelty boundaries

| Work | What it already covers | Required distinction |
|---|---|---|
| [Hierarchical costly-feature acquisition, 2024](https://link.springer.com/article/10.1007/s10994-024-06565-4) | Structured observations, dynamic actions, costs, stopping | Study robustness of acquisition under evidence-preserving graph additions |
| [Relational active acquisition, 2023](https://www.dfki.de/en/web/research/projects-and-publications/publication/14022) | Acquiring relational information | Graph relationships alone are not the novelty |
| [AgentGL, 2026 preprint](https://arxiv.org/abs/2604.05846) | Learned graph-tool exploration | Measure and address failures in the exploration policy |
| [TransXion, 2026 preprint](https://arxiv.org/abs/2604.17420) | AML benchmarking with adversarial graph edits | Isolate query selection from detector evasion |
| [LogicPoison, ACL 2026](https://aclanthology.org/2026.acl-long.252/) | Corruption of graph reasoning paths | Preserve original diagnostic relationships and a feasible retrieval sequence |
| [Adversarial active learning, 2014](https://istc-cc.cmu.edu/publications/papers/2014/aisec14-active.pdf) and [Deception Game, CoRL 2023](https://proceedings.mlr.press/v229/hu23b/hu23b.pdf) | Adversarial querying and learning-aware deception | Do not claim that attacking search or robust information gathering is new |

This is a stronger candidate to investigate, not a verified first contribution.
Complete a focused full-text comparison before using novelty language in a paper.

## 2. A deliberately narrow experimental problem

### Agent and observation protocol

Keep account nodes and directed, timestamped transaction edges. A case begins
with one target transaction and its endpoints. The agent can request incoming or
outgoing history for a discovered account, or stop. The environment controls
discovery, scope, pagination, and the budget; the policy cannot inspect the database.

Retain the initial implementation defaults:

- Evidence cutoff: strictly before the target timestamp.
- Lookback: 48 hours; first 48 hours are history warmup.
- Page size: 50 records, newest first; parallel transactions are preserved.
- Budgets: 2, 4, 8, and 16 requests, with budget zero as a target-only reference.
- Distinct query states: unqueried, complete, empty, truncated, unavailable.
- Account identifiers join records but are not learned identity embeddings.
- Monetary evidence uses exact amounts and separate currencies.

Use complete source availability and fixed-length investigations for the first
study. Add stopping and source outages only after the central mechanism is
established. This prevents several unrelated failure modes from being conflated.

### Controlled perturbation model

Start with static, additive graph perturbations applied before the investigation.
The perturbation may add a bounded number of transaction edges and new branch
accounts. It cannot change the target, delete or rewrite original transactions,
change timestamps or amounts on original edges, reveal labels, alter the tool
implementation, or issue instructions to the agent.

Use added-edge budgets of 0, 2, 4, and 8. A separate 16-edge condition is a stress
test, not the primary result. Log added accounts, added edges, depth, time range,
and amount distribution. Match random controls on these quantities and attachment
points wherever the comparison permits it. Report matching limitations.

For the controlled benchmark, cap relevant account histories below 50 records,
so pagination cannot explain failure. Include matched benign examples and verify
that additions do not change the symbolic target label. Use identifier and record
order randomization to prevent generator shortcuts.

Financial-data experiments are explicitly labeled **synthetic graph stress tests**.
Preserving an original target label is an experimental convention unless the
financial simulator verifies the augmented history's label. Do not describe
unconstrained graph additions as feasible real banking behavior. Reserve economic
feasibility claims for a later validated simulator with balance and timing checks.

### Two types of diagnostic evidence

**Controlled tasks:** construct temporally ordered cycles and converging paths,
with degree/attribute-matched negative cases. The task is detecting a specified
graph property; a motif is not itself ground-truth proof of laundering. Store a
verifiable witness and, for tiny graphs, enumerate valid query sequences to find
the minimum budget needed to reveal it. Protect all witness edges during additions.

**AML classification:** use IBM's supplied laundering labels. Pattern files are
evaluator-only annotations for the subset they cover, not universal explanations
for all positive labels. Use trace replay and a frozen classifier to diagnose
retrieval-related prediction failures without assuming a known causal witness.

## 3. The decisive paired experiment

For every case and perturbation, retain the following runs:

| Run | Purpose |
|---|---|
| Original graph, ordinary policy | Establish baseline search and prediction |
| Matched random additions, ordinary policy | Measure ordinary distraction or graph growth |
| Targeted additions, ordinary policy | Measure the extra effect of strategic placement |
| Modified graph, replayed reference queries | Check that useful evidence remains retrievable within the same budget |
| Modified graph, robust policy | Evaluate the proposed defense |

Replay uses actual tool responses from the modified graph. It never injects the
original evidence directly, bypasses discovery checks, or makes uncharged calls.
A clean sequence that becomes illegal or no longer reaches diagnostic evidence
does not establish a policy-selection failure.

Define two related outcomes:

- **Witness-discovery failure:** on a controlled positive instance with a valid
  budget-feasible witness sequence, the tested policy fails to retrieve any valid
  witness within the budget.
- **Replay-rescuable prediction failure:** the ordinary policy's clean run is
  correct; its modified-graph run is incorrect; and replaying a valid reference
  sequence on that same modified graph restores a correct prediction using the
  same frozen classifier and budget.

The latter is a controlled diagnostic, not a complete causal identification claim.
Publish both its count as a share of all clean-correct cases and its rate within
the replay-eligible subset. Always show eligibility and exclusion counts. Do not
hide failed perturbations or unreachable cases by reporting only successes.

## 4. Methods, baselines, and implementation stages

### Stage A: establish the effect

Implement breadth-first, random, recency-based, and branch-diversity acquisition.
Add a learned clean-data policy after the deterministic benchmark is reliable.
Use the same GNN classifier for policy comparisons; on controlled tasks also use
a symbolic witness verifier to separate search from prediction entirely.

The GNN uses two 64-dimensional message-passing layers, directed transaction
features, explicit reverse relations, target indicators, and query-completeness
state. Only released data enters its graph. Its action head scores valid queries.
Randomized policies are evaluated across repeated draws; identical observations
and fixed random seeds must yield identical choices.

### Stage B: candidate robust method

Use paired original/perturbed training episodes and a frozen classifier. Train
the acquisition policy against bounded additions, rewarding useful evidence
discovery or reduction in classification loss under a fixed query budget. A
reference trace supplies training-only supervision where it remains legal and
successful after perturbation. No reference trace, witness, or label is available
to the deployed policy.

Start with supervised action ranking on these paired traces, then extend to
finite-horizon fitted Q-learning only if longer-horizon planning is empirically
necessary. Include remaining budget and the graph of observed accounts and query
states. Keep representation, classifier, and training-data volume matched to the
clean policy comparison.

Treat paired-trace training as the candidate method, not an established new
algorithm. Compare it directly with ordinary random augmentation and standard
adversarial training. Retain simple branch-diversity and randomized exploration
as serious competitors, even if they win.

### Research baselines beyond the interview

- Structured costly-feature acquisition and a nongreedy ACO-style adaptation.
- An AgentGL-style graph-tool policy, with any adaptations and training differences
  explicitly documented.
- Clean versus randomly augmented versus adversarially trained acquisition.
- Fixed acquisition with a robust classifier, to separate search improvements from
  improvements to prediction.
- A full-authorized-evidence reference, described as a diagnostic reference rather
  than a guaranteed performance upper bound.

Targeted perturbations for the paper must be generated against each defended
policy with matched attacker budgets. Include transfer to unseen policies and
unseen perturbation families. Training and evaluating against the same fixed
perturbation generator is insufficient evidence of robustness.

### Interfaces and artifacts

Preserve the existing `start_case`, `query_history`, and `finish_case` concepts.
Add evaluator-only objects for perturbation manifests, witness certificates,
reference query sequences, and paired-run records. These objects must not be
reachable through policy observations.

Each paired-run record stores dataset/split hashes, case ID, policy/classifier
versions, defender and attacker budgets, seeds, modification manifest, evidence
IDs, complete tool trace, replay eligibility, failure category, and costs.
Keep model checkpoints, raw data, and large runs outside version control.

## 5. Data, evaluation, and checks

### Dataset sequence

1. Controlled graph family for exact witnesses and mechanism isolation.
2. IBM HI-Small for the first public-data experiments; LI-Small for an independent
   simulation from the same generator.
3. TransXion for a second financial generator, after checking data access, license,
   feature provenance, and label semantics.
4. ogbn-arxiv topic prediction for a second application domain, using a documented
   restricted-query adaptation rather than official OGB leaderboard comparisons.

For HI-Small keep only the documented September 1-10, 2022 period; its later
all-positive tail is excluded by time. Use account keys `(bank_id, account_id)`.
Keep the full historical graph when sampling target cases.

Use approximately 60/20/20 chronological target splits with timestamp groups
intact. Separate validation tuning and calibration. Preprocessing and model
training use training data only. Use generator-seed-disjoint splits for the
controlled benchmark and held-out structural sizes for generalization.

### Metrics

Primary mechanism plot: witness-discovery success versus query budget, comparing
original, random-addition, targeted-addition, and defended conditions.

Primary application plot: average precision and recall at fixed review capacity
versus query cost, under the same conditions. Freeze operating thresholds on
validation data. Also report replay-rescuable failures, benign false positives,
calls, returned rows, latency, attacker search effort, and perturbation magnitude.

For a small interview study use a fixed diagnostic cohort and label it as such.
Do not infer population precision from a balanced cohort. The subsequent AML
evaluation includes all eligible positives and 20,000 uniformly sampled negatives,
with sampling weights. Confirm headline paper results on complete eligible test
populations. Count uncertainty at the case/template level for controlled tasks;
report dependence-aware sensitivity analyses for overlapping financial cases.

Use three training seeds initially and five for the paper. Report paired effect
sizes and intervals, not a single best run. A short synthetic financial time span
does not support strong real-world temporal-generalization claims.

### Mandatory tests

- Evidence is strictly historical and future-edge insertion cannot affect a case.
- Identical released observations yield identical fixed-seed actions and scores.
- Identifiers do not become predictive features; renaming preserves behavior.
- Labels, witnesses, attack manifests, and reference traces cannot enter agent state.
- Perturbations preserve original edges and controlled-task labels.
- Replay never bypasses discovery, source scope, pagination, or call accounting.
- All pure-search controlled cases avoid pagination overflow.
- Denied and empty histories remain distinct; each attempt consumes its declared cost.
- Monetary totals remain separated by currency; repeated edges are not collapsed.
- Label-shuffling and generator-shortcut controls are run before substantive claims.

## 6. Schedule and decision gates

The user has 30-50 focused hours before the interview, public data only, and an
initial spending ceiling of $300. No company-data or expert-labeling dependency.

### First 30 hours: a credible failure study

| Work | Hours | Deliverable |
|---|---:|---|
| Closest-work comparison and frozen experimental protocol | 4 | Specific novelty boundary and registered comparisons |
| Controlled graphs, tool boundary, witnesses, replay, and tests | 8 | Auditable paired experiment |
| Baselines and matched distractor conditions | 6 | Initial mechanism results |
| Small GNN and first public-data diagnostic run | 8 | Graph-learning component and measured limitations |
| Figures, three-case viewer, and interview rehearsal | 4 | Clear walkthrough of verified results |
| Total | 30 | Interview-ready study, regardless of whether the hypothesis wins |

At hour 18, inspect deterministic-policy results and implementation validity; a
learned-policy claim is not yet required. At hour 26, inspect the learned policy
and public-data diagnostic results. Reserve hours 31-50 for paired-trace training
(8h), additional seeds and controls (8h), and Colab/documentation (4h), only where
the earlier results justify them.

**Continue the main research direction** if a policy-specific failure survives
matched random additions, same-budget replay, pagination controls, and strong
simple policies on held-out graph instances. As an internal practical threshold,
look for at least a 10 percentage-point targeted-versus-random discovery gap at a
fixed preselected budget in two controlled families. This is a go/no-go heuristic,
not a significance guarantee or predicted effect.

**Revise or stop the novelty claim** if the effect relies on arbitrary record
ordering, impossible graph edits, overwhelming volume, or classifier corruption;
or if a simple policy removes it without a meaningful tradeoff. Preserve those
results. Do not spend the remaining interview time inventing a complicated method
to force a positive result.

### Quarter and following 2-3 months

- Weeks 3-6: paired-trace method, strong acquisition baselines, and analytical
  examples explaining when distraction is avoidable or information-theoretically
  unavoidable under the query interface.
- Weeks 7-10: independent datasets, adaptive attacks against defenses, equal-budget
  comparisons, ablations, and the reproducible course blog/Colab.
- Following 2-3 months: nonfinancial domain, full evaluations, five-seed results,
  closest-work refresh, artifact release, and manuscript preparation.

For the defense, use an internal target of recovering at least half of the
targeted discovery loss while losing no more than two absolute percentage points
on clean discovery success at the same budget. Report the entire tradeoff curve;
this target is not itself a conference acceptance criterion.

Initial cost allocation: $150 compute, $50 optional LLM comparisons, $100 reserve.
No API or GPU spending is required for the initial controlled study. The longer
paper effort is not assumed to fit indefinitely within that initial budget.

## 7. Course, interview, and publication outputs

**Course:** retain Project 1, public data, a substantive PyG implementation, and a
reproducible blog plus Colab. Explain what each control rules out. Allocate figures
to the task, paired replay, discovery-cost curve, clean/robust tradeoff, and failure
analysis. The supplied 2025 rubric weights the final blog 50 points, Colab 25,
and Medium account 5. Confirm current solo eligibility and current deadlines via
course channels; do not treat the attached historical dates as 2026 deadlines.
Submit the private blog draft for course review before public release as directed.

**Interview:** one clean case, its distractor-modified counterpart, and one defense
or failure case. Show actual query paths and measured results. A suitable opening:
"I am studying whether a graph investigator can miss available evidence because
it chooses the wrong search path. I built an evaluation that separates search
failures from prediction failures, then tested methods to improve the search."
Adjust the last clause to what has actually been implemented and tested.

**Publication:** aim at a NeurIPS/ICLR/ICML main-track submission once the mechanism,
method or analysis, and generality are supported. NeurIPS 2027 is a planning target,
not a verified deadline or promised acceptance. A polished application, an attack
on a weak baseline, or a synthetic-only result does not meet the intended bar.

## 8. Current implementation status and immediate order

As of September 23, 2026, the active repository contains the research documents and
IBM HI-Small transaction and pattern downloads with checksums. Implementation will
begin from scratch. The unfinished source scaffold, dependency configuration,
lockfile, Python environment, and empty artifacts directory were moved to the
sibling folder `../CS224W-fraud-detection-machine-learning-with-graphs-scaffold-archive-20260923/`.
Its `ARCHIVE.md` records the original locations. Nothing was deleted. No behavioral
correctness checks, training results, benchmarks, or UI have been completed.

Suggested implementation order: build and test the evidence boundary; implement controlled graphs
and witness verification; add paired replay and deterministic policies; run the
first study; then build the GNN and learned policy. The original one-step
acquisition design remains a baseline. Build the mechanism experiment before a
dashboard or broader agent framework.

Sources: [IBM AMLworld](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
[TransXion](https://github.com/chaos-max/TransXion),
[OGB](https://ogb.stanford.edu/docs/nodeprop/),
[Socratix](https://getsocratix.ai/),
[MTS role](https://www.ycombinator.com/companies/socratix-ai/jobs/JtqwNF7-member-of-technical-staff-ai-agents).
The research papers are linked alongside the claims they constrain above.
