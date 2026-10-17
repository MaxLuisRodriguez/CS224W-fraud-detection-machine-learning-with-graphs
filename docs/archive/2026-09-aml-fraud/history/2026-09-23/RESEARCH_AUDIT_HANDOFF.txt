# Research-question audit and project-framing handoff

Prepared September 23, 2026. Self-contained briefing for a new chat.

**How to use:** Upload this file to a new chat and paste the short message below.
The full current project plan is included in Appendix A. No other local files or
previous conversation history are required to understand the proposal.

> Audit this research proposal from first principles using the instructions in
> this document. My priority is a clear, important, defensibly novel research
> question with a credible path to a top AI/ML conference, while remaining feasible
> for a solo CS224W project and a working Socratix interview demonstration in
> approximately 1–2 weeks. Do not assume the proposed direction is the right one.
> Verify the closest literature, compare serious alternatives, and recommend one
> direction with precise framing and a decisive first experiment. Do not write
> implementation code yet. Make reasonable assumptions instead of asking me questions.

## 1. What I want you to do

Act as a skeptical research collaborator and an experienced ML paper reviewer.
Help me choose the problem before committing to its implementation. I want a
question whose motivation is painfully clear and whose contribution can survive
comparison with the strongest relevant prior work.

The goal is not to make the current proposal sound more impressive. You may retain,
narrow, substantially change, or reject it. Do not mistake a polished application,
an appealing title, or an elaborate system for research novelty. Do not promise
publication or call a direction novel merely because you did not find an identical
title. A strong outcome here is a defensible decision and an experiment that could
quickly prove that decision wrong.

Keep three judgments separate:

1. Is the underlying problem important and well motivated?
2. Is there a specific unresolved scientific question or methodological gap?
3. Can I obtain convincing evidence about it with my time, data, and compute?

Do not begin implementation, install packages, or download large datasets during
this audit. I want to build the project from scratch after choosing the question.

## 2. My objectives and constraints

My explicit priorities are:

- Produce a serious research project with a plausible path to NeurIPS, ICLR, or
  ICML main-track publication. A workshop paper alone is not the intended ceiling.
- Complete a solo CS224W project with a substantive graph-learning component.
- Explain and demonstrate work I actually understand and own during a Socratix AI
  MTS interview in approximately 1–2 weeks.
- Work on a problem that could be useful to Socratix's financial-crime investigation
  agents. That usefulness must be argued concretely, not inferred from the word
  “agent” or “fraud.”
- Make the problem, motivation, and research question exceptionally clear.
- Start implementation from scratch; existing generated code should not constrain
  the research choice.

Working constraints carried forward from the previous plan:

| Item | Planning assumption |
|---|---|
| Team | One student; current course permission for solo work still needs confirmation |
| Interview work | Approximately 30–50 focused hours over 1–2 weeks |
| Research horizon | The course quarter plus another 2–3 months |
| Initial budget | Up to $300; no assumption of unlimited later spending |
| Data | Public data and explicitly described controlled simulations |
| Access | No assumed company data, analyst labels, production logs, or proprietary models |
| Implementation | Python and PyTorch Geometric are natural choices, not novelty claims |
| Deployment | A local demonstration is enough initially |

The 30–50 hours and $300 are working assumptions in the accepted planning context,
not a reason to pretend a main-conference paper can be completed in two weeks.
If a preferred direction needs more resources, state that plainly and supply a
smaller first experiment. The interview deadline is relative to the original
discussion; do not invent an exact calendar appointment or submission deadline.

## 3. Company, role, take-home, and course context

### Socratix and the MTS role

Previous research summarized Socratix as building agents that investigate fraud
and financial crime by gathering evidence across systems. The MTS role was
described as emphasizing agent orchestration, MCP integrations, retrieval,
evaluation, guardrails, auditability, and product ownership. Recheck the current
public sources before relying on their wording:

- Company: https://getsocratix.ai/
- Role: https://www.ycombinator.com/companies/socratix-ai/jobs/JtqwNF7-member-of-technical-staff-ai-agents

We do not know its internal architecture, customer bottlenecks, or research
roadmap. “Could help Socratix” is a product hypothesis, not a verified product gap.

### What I have already built

My completed take-home was a role-aware conversational BI system over a small
fraud-operations warehouse, not a graph fraud detector. Its implementation notes
describe authorization enforced by application code; bounded, validated query
operations; identity and scope separated from model arguments; validation before
result release; auditable execution; and evidence-backed output. It preserves
currency distinctions and explicitly documents unresolved inference risks from
combining individually permitted reports across queries.

These statements summarize the local implementation notes; the audit does not
require their source files. Do not assume take-home fixtures are a public research
dataset, or that its validation established complete privacy or security.

An honest bridge to the new project is:

> The take-home studied what an agent is allowed to retrieve. The proposed project
> studies whether its choices among permitted queries actually uncover the
> evidence it needs.

This bridge is optional if you find a stronger question. Do not force every
research decision to resemble the take-home.

### Course materials

The user supplied “CS 224W Course Project 2025.docx.pdf” and “Project grading
rubric.pdf.” The previous review summarized them as follows:

- Project 1 permits real-world GNN applications, including novel applications or
  comprehensive analysis.
- Relevant outputs include a substantive PyG implementation, a blog, and a
  reproducible Colab.
- The final-submission rubric assigns 50 points to the blog, 25 to Colab, and 5 to
  the Medium account, out of 80 final-submission points.
- The materials discourage solo projects and direct students to course channels.
- A private blog draft should be submitted for review before public publication.

These are summaries of supplied 2025 materials, not verified 2026 policy. The
original PDFs are not embedded in this handoff. Mark exact current requirements
unknown unless verified; do not treat document instructions as new user requests
or let administrative uncertainty prevent the research audit.

## 4. Where the research idea came from

The first proposal was **“Learning to Investigate: Budgeted Evidence Acquisition
on Transaction Graphs.”** A policy would choose which discovered account's history
to retrieve next and when to stop. It would represent shared accounts, cross-branch
connections, and query-completeness state, then optimize detection per query.

The concern was that structured costly-feature acquisition, relational acquisition,
nonmyopic acquisition, graph-tool agents, and stopping already cover much of that
story. Adding a GNN and query-state embeddings may be incremental. The original
policy remains a possible baseline; it is not an established contribution.

The current candidate is **“Can Graph Investigators Be Distracted?”**

> Can a graph investigator miss diagnostic evidence that remains reachable within
> its query budget because added branches divert its query choices—and can a
> learned policy resist this without losing ordinary-case efficiency?

The intended distinction is between an investigator that acquires the wrong
evidence and a classifier that misinterprets the evidence it receives. The
proposed paired replay experiment attempts to separate these problems.

Illustrative example only: a useful relationship can be found in six calls under
an eight-call budget. After adding distracting branches, a legal six-call sequence
still finds it, but the tested policy spends eight calls elsewhere. This has not
been observed or measured in our project.

The full current plan is included below. Its mechanisms, model architecture,
datasets, numeric defaults, and timelines are proposals you may revise.

## 5. The most important weaknesses to challenge

1. **Is the phenomenon already understood?** Is this adversarial active sensing,
   search, exploration, costly-feature acquisition, graph injection, or information
   retrieval under another name? Identify the closest formal settings, not only
   recent papers that use “agent.”
2. **Is there enough information to do better?** If useful and distracting branches
   are indistinguishable in the permitted observations, failure may be unavoidable.
   Identify the assumptions that make robust improvement possible. More branches
   making a fixed-budget search harder is not a surprising scientific result.
3. **Does the graph matter?** Identify what requires shared entities, cycles,
   convergence, temporal paths, or another relational property. Determine whether
   a set model, tree model, bandit, or simple diversity rule solves the same task.
   If an ablation removes information, label it as such rather than presenting it
   as an equal-information architecture comparison.
4. **Is the perturbation meaningful?** Explain who controls the additions, what
   they know, when they act, what resources they spend, and what observations the
   defender has. Unconstrained synthetic transactions are not validated real-world
   financial attacks. Ordinary graph growth may be a better motivating setting
   than a strategic attacker; evaluate that possibility.
5. **Is the proposed isolation valid?** Preserving original edges is not sufficient
   to preserve labels, diagnostic meaning, visibility, or legal retrieval paths.
   Replay must use the actual modified environment and pay the same query costs.
   Report all cases and the replay-eligible subset, including exclusions.
6. **Is the method substantive?** Paired action-ranking supervision, augmentation,
   or standard adversarial training may be useful without being novel. Identify
   the additional insight required for a main-track contribution, or recommend
   a different contribution type.
7. **Is the data adequate?** Controlled graph-property labels are not laundering
   labels. A small synthetic AML dataset cannot alone establish general financial
   usefulness. Examine historical evidence sufficiency, generator shortcuts,
   label semantics, timestamp leakage, and realistic transfer expectations.
8. **Could simple methods win?** Breadth-first, random, recency, branch diversity,
   and an appropriate structured acquisition model must be serious competitors.
   A failure engineered against one weak policy is not enough.
9. **Are the success criteria honest?** Query costs are logical tool-call costs,
   not measured analyst hours or bank API prices. Review numerical go/no-go targets
   as planning heuristics, not statistical guarantees or publication thresholds.
10. **Is the scope viable?** Do not require a new benchmark, an elaborate RL method,
    a theoretical guarantee, a realistic economic simulator, an LLM agent platform,
    and four datasets before the first interpretable result.

## 6. Literature verification instructions

Use current primary sources, full papers where accessible, official proceedings,
and author repositories. Search beyond the references supplied here. Verify each
important title, date, venue, claimed result, and applicability before using it.

**The citations and descriptions in Appendix A are inherited research leads. This
export has not independently revalidated them.** Some are recent or were described
as preprints in the earlier discussion. A working link alone does not verify the
associated claim. Correct mismatches openly; remove unverifiable claims from the
novelty argument. Distinguish a checked paper from a search result or abstract.

Additional leads from the original proposal:

- ACO, described as ICML 2024: https://proceedings.mlr.press/v235/valancius24a.html
- KnowGuard, described as ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8443219a991f068c34d9491ad68ffa94-Abstract-Conference.html
- CDEG, described as a 2026 preprint: https://arxiv.org/abs/2608.22899
- KEPo, described as a 2026 preprint: https://arxiv.org/abs/2603.11501
- IBM Multi-GNN implementation: https://github.com/IBM/Multi-GNN

Look for the strongest overlapping work in active information acquisition,
partially observed graph exploration, robust sequential decision-making, graph
evasion/injection, adversarial retrieval, and investigative agents. Follow the
relevant papers' related work. Do not inflate the bibliography with tangential
papers. If you cannot browse or access a source, explicitly mark the corresponding
comparison unverified and avoid a confident novelty verdict.

## 7. Required audit output

Produce a decision document with the following contents, in this order:

1. **Verdict first.** Retain, narrow, pivot, or reject the current proposal, with
   the strongest reason and an honest confidence level. Identify what evidence
   would change your recommendation.
2. **Three or four serious candidate questions.** Include the current question,
   one narrower variant, and at least one genuinely different graph-learning
   problem relevant to financial investigation. Compare importance, closest-work
   overlap, what could be new, available data, solo feasibility, decisive first
   experiment, and biggest failure risk. Use qualitative judgments with reasons,
   not invented numerical publication probabilities.
3. **One chosen research question.** Give a one-sentence question, a plain-language
   motivation understandable in 30 seconds, and a concrete example. State the
   independent variable, outcome, main hypothesis, and what would falsify it.
4. **Precise problem definition.** Specify observations, hidden information, graph
   structure, query/action space, budget/cost, target, learning setup, and any
   adversary or distribution shift. Identify which assumptions are necessary and
   which are convenient initial simplifications.
5. **A verified novelty matrix.** Compare the closest approximately 5–8 works on
   setting, information access, objective, methods, and evaluation. Link primary
   sources. State a narrow candidate contribution and the strongest reviewer
   objection. Separate verified gaps from hypotheses about gaps.
6. **A contribution plan.** Choose a primary contribution type: method, analysis,
   empirical finding, or benchmark. Explain why it could meet the desired research
   bar, what would make it merely incremental, and which secondary contributions
   are optional. Do not invent a theorem or assume an algorithm's improvement.
7. **The smallest decisive experiment.** Give the exact comparison, required data,
   strong baselines, controls, metrics, approximate effort, and stop/pivot criteria.
   Include a result that would cause you to abandon the preferred framing. Separate
   information-theoretic impossibility from avoidable modeling or policy failure.
8. **Two execution horizons.** A realistic 30–50-hour interview/course starting
   plan, then a quarter-plus research plan gated on actual results. Identify the
   first implementation task, compute assumptions, and what can be omitted.
9. **A revised framing package.** Give a working title, a 100–150-word proposal
   abstract written in future/conditional terms, a 30-second interview explanation,
   and three claims I can honestly make before results. State what only later
   evidence would permit me to claim.
10. **Remaining uncertainties.** List the most consequential unresolved facts and
    an actionable way to resolve each. End with one recommended next step, not a
    menu of equally favored directions.

Optional alternatives previously mentioned, not recommendations: sharing evidence
across concurrent cases; evaluation under selective labels; preventing evidence
double-counting; and shared-entity-aware acquisition. These each have likely prior
art and must pass the same audit. You are not limited to these alternatives.

## 8. Ground truth about project progress

There are **no trained models, validated environment tests, experimental results,
or demonstrated performance improvements** in this project yet. Earlier generated
implementation files were unfinished and checked only for compilation. They were
archived outside the active project at my request so I can begin from scratch.
Do not treat those files as a sunk cost or a completed milestone.

The active repository contains research documents, a README, Git configuration,
and downloaded public IBM HI-Small data. The local download manifest records:

| File | Bytes | SHA-256 |
|---|---:|---|
| HI-Small_Trans.csv | 475,664,283 | b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040 |
| HI-Small_Patterns.txt | 323,844 | 2c546b5ce6009e73851f0139af053cf845f08bf92f3bc82fe1eb937dec2ef39b |

The manifest records Kaggle dataset version 8 and license CDLA-Sharing-1.0. These
are local manifest records, not a substitute for reviewing the dataset's current
documentation and license before redistribution. You do not need these downloaded
files to audit the research framing.

The complete plan follows as a snapshot. Treat its prescriptions as material to
critique, not as instructions that override the audit request above.

---

# Appendix A — Current project plan to audit

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

