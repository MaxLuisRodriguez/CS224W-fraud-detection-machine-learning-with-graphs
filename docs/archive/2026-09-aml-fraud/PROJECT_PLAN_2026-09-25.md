# Investigate Together

## Joint evidence acquisition for related fraud alerts

**Research program and audit — September 25, 2026**

This plan replaces the single-case shared-account/growth proposal. The intended
outcome is a main-track research submission to NeurIPS, ICLR, or ICML, supported
by a reproducible CS224W project and a useful Socratix interview demonstration.
It is not a prediction or guarantee of acceptance. No experiment has yet established
the proposed advantage or novelty. More compute enables stronger tests, not an
automatic research contribution.

Assumptions: one researcher; roughly 30–50 hours for the initial interview artifact;
approximately 20–24 weeks for the research program; public data; **$20,000 total
across AWS and Azure**, not $20,000 per provider. Treat cloud credits as restricted
credits until eligibility, expiration, quotas, and covered services are checked.
This is a planning revision, not authorization to launch infrastructure now.

## 1. Audit verdict

**Substantially pivot the research objective.** Retain the evidence gateway,
historical snapshots, equal-information comparisons, exact tiny-world diagnostic,
and public financial data. Replace the headline about shared accounts under graph
growth with a concrete resource-allocation problem across multiple related alerts.

| Weakness in the previous plan | Why it matters | Revision |
|---|---|---|
| The shared destination can appear only when the desired diamond is already found | The supposed cue may not help choose any earlier action | Shared accounts connect multiple initially visible alert endpoints or become shared in retrieved evidence; their histories can affect several still-undecided cases |
| Finding a motif is not detecting laundering | A perfect search experiment could remain disconnected from the application's labels | Primary application outcome is classification/ranking of supplied laundering labels across an alert queue |
| GNN action selection plus a standard optimal reference is incremental | It does not establish a new method or general insight | Test a specific failure of case-local acquisition and a joint continuation-value model, against strong global acquisition/planning baselines |
| An artificial query budget may manufacture a benefit | A cheap bulk retrieval could erase the operational premise | Report logical calls, rows, bytes, latency, and bulk-query controls; keep operational claims conditional |
| Extra graph growth can simply make search harder | Loss need not be avoidable | Retain exact same-observation references on finite distributions; move graph growth to a stress test |
| A $300/44-hour plan was serving as a paper roadmap | A pilot and a main-track research effort are different deliverables | Separate the 50-hour prototype from a staged, 20–24-week research program |

The new plan also has a serious novelty risk: active inference and cohort-level
acquisition already exist. A queue, a cache, a GNN, or their combination is not
sufficient. The decisive test is performance beyond a strong global planner with
the same evidence sharing, under a plausible evidence interface.

## 2. Final question and concrete motivation

> **Can jointly choosing evidence requests across related fraud alerts improve
> money-laundering detection under a fixed investigation budget, beyond
> investigating alerts individually with the same shared cache?**

**Publication research question:** Can we learn which unresolved cases an evidence
request could help, including through later requests, and use that information to
match dense joint planning with substantially less computation as the queue grows?

The first question establishes application value. The publication question tests
a narrower candidate mechanism: adaptive computation of action–case influence.
It must also beat efficient scalar joint models; an expensive dense architecture
is not an acceptable straw-man baseline.

**Why the graph matters:** transactions connect accounts; alerts refer to specific
transactions; a history query returns records that may be relevant to multiple
alerts. The usefulness of a request depends on the whole unresolved queue, the
records already acquired, and what the request can make available next.

“Shared account” now has an unambiguous meaning: the same bank/account identity
occurs in the evidence or endpoints of more than one alert. It is not a joint
bank account, a guessed common owner, or a learned account-ID embedding.

### A hypothetical investigation

At a fixed review time, the queue contains alerts for A → X, B → X, and C → Y.
X is already visibly shared by the first two alerts. The investigator has four
history requests for the whole queue.

1. Querying X may reveal a transfer to an initially unknown Z. That fact alone
   may barely change either alert's risk.
2. Querying Z may reveal a rapid redistribution pattern which, together with
   the first records, changes the predictions for both A → X and B → X.
3. A case-local policy might prefer an immediately informative request about A,
   because the X-then-Z sequence is expensive when valued for A's case alone.
4. A joint policy can value the sequence's benefit across both cases. A cache
   does not fix a sequence the local policy never chooses to start.
5. If X is a legitimate merchant, its shared status may be uninformative. Benign
   merchants, payment processors, and shared payroll-like structures are essential
   counterexamples. A “query the most shared account” heuristic must be competitive.

This is a hypothesis, not a guaranteed advantage or an observed result. In the
synthetic financial application, A → X and B → X are positive only if the dataset's
ground truth marks them as laundering. Neither sharing X nor a redistribution
pattern proves crime. The records change estimated risk; they do not establish
legal intent or beneficial ownership.

### Socratix connection

The potential product is a scheduler for evidence acquisition across a case queue,
plus an evaluation harness that measures its benefit. It complements scoped tool
execution and auditability from the take-home. No company data or actual Socratix
weakness is assumed. Its public [MTS role](https://www.ycombinator.com/companies/socratix-ai/jobs/JtqwNF7-member-of-technical-staff-ai-agents)
motivates evidence gathering, tool integration, and trustworthy evaluations; the
existence of this particular queue-level bottleneck still needs validation.

## 3. What the paper would have to contribute

**Primary contribution:** a scalable method for sequential acquisition with shared
evidence and several prediction targets, supported by an analysis of when
case-local planning misses useful acquisitions.

The proposed paper should establish three claims, conditionally:

1. **Failure mechanism:** even with perfect caching, local acquisition can overlook
   sequences whose useful payoff is spread across multiple related cases.
2. **Method:** adaptively computing potentially useful action–case contributions
   preserves delayed shared-evidence value while improving the detection–runtime
   frontier over dense, scalar, and ordinary sparse joint acquisition models.
3. **Generality:** results hold across two financial generators, unseen queue sizes
   and overlap patterns, and a nonfinancial relational acquisition task.

The first claim alone is unlikely to be new. The model and evidence must distinguish
the work from existing collective inference, nonmyopic value-of-information, and
cohort acquisition. No theorem or performance improvement is promised in advance.

**Do not pursue the method-paper claim** if ordinary global greedy acquisition or
a straightforward existing planner solves the problem at comparable runtime.
A rigorous negative result can still complete the course project.

## 4. Prior work and remaining novelty risk

**Literature correction:** the dense common-policy value-head model in section 6
is an inherited baseline, not the proposed new algorithm. Reward decomposition
and successor features already cover the central construction; recent Bellman
risk-to-go acquisition closely overlaps its training recipe.

The [detailed September 25 audit](RELATED_WORK_AUDIT_2026-09-25.md) records current
papers, inspection scope, overlap, required adaptations, and unresolved novelty
checks. It is part of this plan. Neither this audit nor the table below constitutes
an exhaustive novelty clearance or a reproduction of the cited results.

| Work | Overlap and inspection | Required comparison/distinction |
|---|---|---|
| [Effective Label Acquisition for Collective Classification, KDD 2008](https://linqs.org/assets/resources/bilgic-kdd08.pdf) | Primary paper: querying labels to improve predictions across related nodes | Our tools acquire raw evidence and reveal topology, not ground-truth labels; collective benefit itself is established |
| [Optimal Nonmyopic Value of Information, IJCAI 2005](https://www.ijcai.org/Proceedings/05/Papers/1154.pdf) | Primary paper: conditional observation plans in graphical models | Multi-step information value and planning theory are not new |
| [Cohort-Based Active Modality Acquisition, May 2026 version](https://arxiv.org/html/2505.16791v4) | Formulation inspected: allocating acquisitions across a prediction cohort | A cohort budget is not novel; test action reuse across targets and sequentially revealed queries |
| [Hierarchical costly-feature acquisition, 2024](https://link.springer.com/article/10.1007/s10994-024-06565-4) | Earlier primary-source check: structured sequential acquisition | Match structured data, dynamic actions, and metadata access |
| [ACO, ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html) | Proceedings record checked; nongreedy feature acquisition | Adapt a strong nonmyopic acquisition baseline to the joint objective |
| [BRiG-AFA, August 2026 preprint](https://arxiv.org/html/2608.02305) | Sections 3–4: frozen predictor and budget-specific fitted risk targets | Mandatory scalar joint-risk baseline; Bellman training is inherited |
| [Pathwise AFA, May 2026 preprint](https://arxiv.org/html/2605.05511) | Sections 2–3: relaxed acquisition policy training | Compare on controlled fixed-action tasks; document graph-action adaptation |
| [Successor Features, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/350db081a661525235354dd3e19b8c05-Paper.pdf) | Common-policy component returns | Per-alert value decomposition is not new |
| [Deep Coordination Graphs, ICML 2020](https://proceedings.mlr.press/v119/boehmer20a.html) | Graph-factorized value computation | Sparsity and graph value factorization alone are insufficient |
| [Zero-Shot AFA via LLM-Elicitation, June 2026 preprint](https://arxiv.org/html/2606.18933) | Top-k shared-budget decisions and selective computation | Compare relevant pruning mechanisms before claiming adaptive influence novelty |
| [PEGE, June 2026 version](https://arxiv.org/html/2601.16233v2) | Graph-observation formulation inspected | Include sampled graph-completion planning; do not claim partial topology is new |
| [AgentGL](https://arxiv.org/abs/2604.05846v2) | Earlier author-record check: graph-tool RL | A tool-using graph policy is not a contribution by itself |
| [Toward Auditable Fraud Detection, July 2026 preprint](https://arxiv.org/abs/2607.19266) | Abstract inspected: graph/agent components do not uniformly improve detection | Test simple predictors and simulator shortcuts; do not infer value from convincing explanations |

Also review [Informational Substitutes](https://arxiv.org/abs/1703.08636) before
claiming a new theory of evidence complementarity. Adaptive diminishing returns
cannot simply be assumed for joint classification loss.

**Remaining unverified gap:** whether adaptive refinement of action–case influence
can preserve useful delayed effects at lower total computational cost when evidence
reveals new cross-case connections. Existing acquisition pruning and factored-value
methods may already supply the solution. Differences in task formulation are a
reason to test, not proof of novelty.

Novelty gate before major spending: reproduce the relevant formulations, map their
observation access and objective onto this setting, and implement the strongest
straightforward adaptation. If it already supplies the proposed algorithm, change
the contribution or stop. Do not rename an existing method.

## 5. Task definition and fair evidence access

### Queue and temporal scope

An episode is one queue Q of N candidate transaction alerts at a common review
time tau. Each alert has its transaction metadata, a fixed screening score, and
an evaluator-only laundering label y_i. Begin with N=32; use N=128 as the principal
application setting and N=512 for size transfer.

This is explicitly **retrospective alert review**, replacing the old requirement
that every record precede each individual target transaction. Queries return only
records timestamped strictly before tau. Some may follow a target but precede the
review time. Never describe these experiments as real-time transaction blocking.

Initially use targets from the preceding 24 hours and a 48-hour history window.
Record event-time and observation-time assumptions: public logs generally lack
ingestion timestamps, so assuming immediate availability is a limitation. Test
delayed availability later. Keep an identical cutoff and scope for all policies.

### Queries and cache

An action requests an incoming or outgoing history page for a discovered account:
account, direction, window, cutoff, and cursor identify the request. Return at most
50 records in deterministic newest-first order, with complete/empty/truncated
status. Include unavailable sources only in a later stress test. Account identity
is the bank/account pair. Preserve direction, parallel transfers, exact display
amounts, and separate currencies.

The initial discovered set is the union of all alert endpoints. New accounts enter
only through returned records. Do not reveal induced edges, hidden degrees, hidden
neighbors, pattern membership, or laundering labels.

**Every substantive baseline receives the same global evidence store and cache.**
Returned records can update all alert predictions. Repeated identical requests
are removed from the action set. Cached pages cost zero retrieval calls for every
method; memory lookup, processing, and model inference still count toward runtime.

Records returned through different incoming/outgoing pages are deduplicated as
evidence, but both distinct requests incur their declared cost. Report duplicate
rows and useful unique records. Reset caches between primary evaluation episodes;
add an equally warm cache for all policies as a sensitivity study.

Sharing is only within the same authorized scope and temporal snapshot. Different
tenants, scopes, cutoffs, or page versions are not interchangeable cache entries.
The first study uses one common scope; it makes no cross-bank access assumption.

### Query cost and bulk access

Primary budget B counts unique history-page requests. Evaluate B=0, N/4, N/2, N,
and 2N, with integer rounding fixed in the manifest. Also report frontiers under
returned-row caps and measured latency. Do not claim logical calls are dollar costs.

Include a bulk-history baseline for all initial endpoints and an all-reachable
authorized-evidence diagnostic with explicit row/runtime cost. If the actual
deployment permits cheap bulk access, this may be the best solution. Report that
outcome rather than making its interface artificially expensive.

### Labels, queues, and selection

Construct queues using a frozen training-only screening model and a threshold
chosen on validation data, then split retained alerts chronologically into batches.
Queue formation must never inspect test laundering labels or hidden ring IDs.
Report screen recall, queue prevalence, and end-to-end recall; investigation cannot
recover positives excluded by screening.

Use naturally occurring queues for headlines. Create overlap-enriched and
overlap-depleted diagnostic cohorts using initial observable endpoints only,
label them as interventions, and do not treat them as population samples. Deduplicate
target transaction IDs. A shared legitimate processor must not be interpreted as
a crime ring. Keep isolated alerts in the evaluation.

## 6. Objective and proposed method

### Common predictor and objective

Train a PyG partial-observation risk model f_phi on diverse training trajectories,
then freeze it for the first acquisition comparison. For each alert it outputs
p_i(h) given the released graph and query state. All acquisition policies use this
same predictor. Refit predictors only in a separately reported end-to-end study.

Use mean Brier loss for the initial bounded training reward:

L(h,y) = (1/N) sum_i (p_i(h) - y_i)^2.

The acquisition objective minimizes expected terminal L subject to cost <= B.
At training time, r_t = L(h_t,y) - L(h_{t+1},y); with undiscounted finite-horizon
returns this telescopes to the final improvement. Labels are never observable
to the inference policy. Brier loss is a tractable surrogate, not the headline
fraud metric; report whether optimizing it improves recall at review capacity.

Use disjoint predictor-training and policy-reward folds initially, and cross-fitting
for the larger study. This reduces misleading rewards from a memorized predictor.
All preprocessing and calibration are training/validation-only.

### Inherited dense baseline: joint continuation-value learning

Maintain an observed heterogeneous graph with account nodes, alert nodes, and
legal query/page nodes. Link alerts to their transaction endpoints, accounts by
released transfers, and actions to their accounts/query state. Include remaining
budget, observed timestamps, and costs. Do not learn global account-ID embeddings.

A three-layer, width-128 GNN supplies account/action representations; alert-specific
representations attend to actions and a global queue summary. An action–alert head
q_i(h,a,b) estimates the future loss reduction for alert i after action a followed
by **one shared future queue policy**. The action score is their mean:

Q(h,a,b) = (1/N) sum_i q_i(h,a,b).

The important distinction is the continuation: summing independently optimal
single-case values can assume incompatible uses of the same remaining budget.
All per-alert targets here must use the same next action, selected by the joint
score. A finite-horizon fitted-value target is:

q_i(h,a,b) <- r_i + q_i_target(h', a_star, b-c(a)),
where a_star = argmax_a' mean_j q_j_online(h',a',b-c(a)).

Here r_i = (p_i(h)-y_i)^2 - (p_i(h')-y_i)^2 is that alert's
training-only loss improvement, before averaging across the queue.

At zero remaining budget the continuation is zero. Legal actions and affordable
costs are masked. Initial experiments use unit page costs; later heterogeneous
costs use the actual remaining-cost constraint, not an arbitrary division of a
long-horizon value by cost.

This construction uses established reward decomposition and common-policy component
returns. Treat it as a dense baseline. Also train a scalar queue-risk model using
BRiG-style backward fitting and a scalar budget-conditioned GNN; either may already
solve the task without an action-by-alert output matrix.

### Candidate mechanism: adaptive computation of action–case influence

For each legal request, a cheap router selects alerts whose future loss may be
affected. It uses released graph structure, alert/action embeddings and remaining
budget. A residual head estimates the signed aggregate contribution of omitted
alerts. Refine promising or ambiguous action rankings by evaluating more alert
contributions, falling back to dense evaluation when necessary.

Update routing after every acquired page so newly connected cases can enter the
calculation. Keep a global route for indirect effects; observed disconnection is
not proof of zero future influence. Route on predicted delayed return rather than
only immediate risk changes. Train routing and residual targets using dense
training trajectories, with held-out calibration and no evaluation labels.

This is a concrete research candidate, **not yet a verified novel algorithm**.
Learned residual intervals are empirical; do not call them certified bounds.
Contributions can be negative. Generic top-k attention, sparse message passing,
and a trivial approximate-greedy error bound are not sufficient contributions.

Compare dense heads, scalar models, low-rank heads, static neighborhoods, fixed
top-k routing, random routing, and adaptive refinement. Include routing overhead,
predictor recomputation and candidate construction in end-to-end latency. If the
scalar model matches the result faster, reject the proposed mechanism.

The detailed audit specifies remaining comparisons with acquisition pruning and
factored planning. Complete those checks before presenting this candidate as new.

### Training and planning safeguards

Use a mixture of random, breadth-first, coverage, and current-policy trajectories
in training worlds. Grow the training set with on-policy rollouts to address action
coverage. Never train on held-out episodes or their counterfactual rewards.

The exact tiny-world reference and any scenario planner must group worlds with
identical observations before choosing future actions. Choosing a different future
action for each hidden world creates clairvoyance. Training losses may use known
labels, but action selection and inference must remain observation-conditioned.

On public data, direct value learning does not require generating fictitious bank
records. A generative planner is a comparator, not a prerequisite for the initial
method. Imagined outcomes from any planner are private planning state, never
released as factual evidence or inserted into the audit trace.

## 7. Baselines that could defeat the proposal

| Comparator | Purpose |
|---|---|
| Target-only LightGBM and target-only GNN | Does retrieval add predictive value? |
| Bulk endpoint histories / all authorized evidence | Is selective querying operationally necessary? |
| Random/BFS/recency with global cache | Basic acquisition controls |
| Highest-risk-first or uncertainty-first case selection, with shared cache | Strong ordinary case workflow |
| Best single-case acquisition plus shared cache and adaptive global budget allocation | Is joint value learning better than local optimization with reuse? |
| Most-shared-account / weighted coverage / merchant-aware overlap heuristic | Does a simple reuse rule solve the problem? |
| Global greedy expected loss reduction, updating every alert | Is the benefit just immediate cohort utility? |
| Sum-of-single-case-values and maximum-single-case-value schedulers | Does incompatible local continuation explain a real failure? |
| Identity-aware set/attention policy with the same joint reward | Is graph representation helpful beyond equal-information attention? |
| Scalar BRiG-style joint-risk model and scalar budget-conditioned GNN | Does standard global value learning already solve the task efficiently? |
| Dense common-policy reward decomposition | Is adaptive influence computation useful beyond the inherited construction? |
| Fixed top-k, low-rank, neighborhood and random routing | Is adaptive refinement better than ordinary compression? |
| Pathwise AFA on fixed-candidate controlled tasks | Does a recent alternative training method close the gap? |
| CAMA-inspired acquisition adapted to change multiple scores per request | Compare against cohort allocation rather than only per-case methods |
| ACO-style nonmyopic adaptation and PEGE/POMCP-style joint scenario planner | Compare against serious sequential planning |
| Bounded graph-tool LLM on a declared subset | Interview relevance and comparison; not the main scientific baseline |

Published methods require explicit adaptations; do not call a rough imitation a
faithful reproduction. First reproduce a small author-provided result where feasible,
then document changed observations, actions, rewards, training, and compute.

No method is denied identity links, query completeness, cached records, initial
alerts, or temporal metadata supplied to ours. A local objective is an algorithmic
restriction, not restricted evidence access. Local policies' predictions still
update from the global evidence. Give global methods matched tuning budgets.

## 8. The decisive first experiment and analytical work

Build finite queue worlds with 2–8 alerts, a small action set, and at most 64 complete
worlds per initial observable configuration. Specify priors, labels, query outputs,
and costs explicitly. Compute the optimal observation-constrained policy by belief
state dynamic programming; the oracle knows the prior, not the realized world.

Vary two factors separately:

1. Evidence is private to each case versus useful to several cases.
2. Evidence is immediately useful versus useful only after a follow-up request.

Include benign high-overlap hubs, unrelated alerts, uninformative shared queries,
and exchangeable choices with unavoidable uncertainty. Query caching is identical
in every cell. The hypothesis is an interaction between reuse and delayed benefit,
not merely that more overlap makes acquisition cheaper.

The first comparison is exact reference versus the strongest local scheduler,
global greedy, coverage, and short-horizon planning. The toy worlds are algorithmic
mechanism tests, not claims about financial behavior. A cheap counterexample alone
does not establish a paper contribution.

Analytical agenda:

- Characterize when case-local value accounting loses joint opportunity despite
  caching, and when it is equivalent to an appropriate joint solution.
- In disjoint, independent components, account for optimal global budget allocation;
  do not falsely claim equal per-case budgets are optimal.
- Analyze estimation and truncation errors of the proposed factorization on a
  clearly restricted class, if a useful result exists. Known Bellman-error bounds
  or standard value-of-information arguments must be credited, not repackaged.
- Test whether a strong observed-data overlap statistic predicts the advantage
  before spending the budget. Do not use hidden true relevance as that statistic.

Formal proofs are a possible research output, not promised results. If only a
standard counterexample and ordinary fitted-Q learning remain, the contribution
must come from a substantial new empirical finding or the main-track claim fails.

## 9. Data program

### A. Controlled mechanism distributions

Use exact worlds above and two separately designed scalable generators. One can
model correlated cases with shared tests; another can model graph discovery with
late-revealed common intermediaries. Include legitimate lookalikes and disjoint
controls. Split on template families and generator seeds, not just node IDs.

### B. IBM AMLworld: initial financial application

HI-Small transaction and pattern files are already present locally. Their manifest
records dataset version 8 and CDLA-Sharing-1.0. The transaction CSV header was checked
in this audit: it has two columns named “Account.” Ingestion must explicitly assign
sender-account and receiver-account by their documented positions, and retain bank
IDs as strings, including leading zeros. Never allow a duplicate-name dictionary
reader to silently overwrite one endpoint.

Before policy training, verify the [dataset documentation](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
license, time span, reported all-positive late tail, and pattern coverage against
the actual data. The web page did not expose its full description in this audit;
the earlier date/tail claims remain to be checked. Exclude shortcut periods using
a documented time rule rather than label-dependent test filtering.

Audit at least three things before accepting the task: historical evidence improves
a strong predictor; natural queues exhibit useful overlap rather than only common
merchants; and meaningful labels remain at the fixed review time. Test 0/6/24-hour
review delays and 24/48-hour history windows using validation only, then freeze.

HI/LI are simulations from the same generator. Use larger IBM variants only after
small-data gates; more rows do not create independent evidence of generalization.

### C. TransXion: independent financial generation process

The [author repository](https://github.com/chaos-max/TransXion) reports a year-long
synthetic transaction dataset and profile files. Its paper and repository have
changed over versions. Pin paper version, code commit, data hash, and permitted use
before experiments; do not mix version-specific claims.

Audit profile features for direct generator-label shortcuts. Use transaction-only
inputs as the main cross-generator comparison. Profile acquisition is a separate
extension only after provenance and availability are established. Publicly visible
files do not by themselves establish redistribution rights. If licensing/access
is unsuitable, use [AMLSim](https://github.com/IBM/AMLSim) as a documented fallback;
do not count two IBM files as independent generators.

Evaluate both within-generator training and transfer with thresholds frozen before
the target test. Report any target-domain retraining as a separate adaptation result.

### D. Nonfinancial generality

Use an explicitly restricted-observation adaptation of [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/).
Targets are paper venue labels. Querying discovered author/paper neighborhoods can
reveal records useful to multiple paper targets. Return no venue labels through
tools; historical training labels are used only in fitting. Preserve supplied time
splits and restrict publication-dated edges appropriately. State any relation
availability assumptions where timestamps are absent.

This is an artificial access-cost study on a public heterogeneous graph, not an
official OGB leaderboard result or an operational scholarly-search claim. If it
does not expose meaningful shared acquisition, reject it rather than manufacture
gains. It tests generality, not financial realism.

## 10. Evaluation and ablations

### Primary application outcomes

Measure laundering recall at a fixed per-queue review capacity of ceil(0.1N), versus
total evidence budget. Pool counts over queues and report capacity explicitly.
Also report precision at that capacity, AP, Brier loss/calibration, and end-to-end
recall including screening. Do not average undefined positive-case recall in empty
queues; publish zero-positive queue counts and false-review burden.

No claim of real dollars saved: label-weighted economic costs are sensitivity
analyses unless grounded in an external operational source.

The engineering target is **at least 20% fewer unique page requests at comparable
recall** (within one absolute percentage point) versus the strongest matched
baseline, or a clear recall improvement at matched cost. Require comparable row
and runtime results. This target is not a prediction or conference acceptance bar.

### Statistical protocol

Use chronological target splits with intact timestamp groups and disjoint queues.
If history windows overlap split boundaries, distinguish lawful earlier evidence
from training leakage and run an embargoed sensitivity analysis. Fit every model,
screen, calibrator, and stopping/operating rule without test labels.

Use five seeds for headline learned results and three during development. Cluster
uncertainty by time blocks and connected-case groups; include sensitivity to block
size and very large components. The short IBM time span cannot provide many
independent temporal replications, which motivates longer and independent data.

Do not construct headline queues by taking all positives and convenient negatives.
That alters the allocation problem. Prefer full eligible queues; if subsampling is
necessary, sample whole queues independently of their labels and retain inclusion
probabilities. Report exact targets and compute for every headline comparison.

Preregister the primary dataset, queue rule, budget, capacity, and comparison before
the final test. Use held-out account groups and future periods as distinct tests;
do not imply temporal generalization guarantees unseen-account generalization.

### Essential ablations

For the conditional efficiency contribution, predeclare an engineering target of
2x lower end-to-end acquisition-decision latency with at most one absolute
percentage point loss in recall at fixed review capacity, at matched retrieval
cost, against the strongest dense comparator. Separately require a useful frontier
improvement over efficient scalar and heuristic competitors. Report paired
uncertainty; these targets are not significance tests or predicted results.

Measure queue-size scaling, dense overlap, newly revealed bridges, residual error,
fallback frequency and total prediction-plus-routing runtime. Speeding only a
minor scoring kernel does not establish a system-level efficiency improvement.

- Full shared cache for everyone versus a deliberately no-cache diagnostic.
- Joint versus local acquisition objective with the same frozen risk predictor.
- Global immediate value versus shared multi-step continuation.
- Shared versus independently optimized continuation targets.
- Graph versus identity-aware attention with equal observations and supervision.
- Natural overlap, legitimate hubs, isolated cases, and unseen queue sizes.
- Clean versus growth-shifted queues; growth is no longer the central claim.
- Warm versus cold caches, page sizes, response-row costs, and bulk availability.
- Runtime-matched and retrieval-budget-matched planning comparisons.

### Correctness and falsification

Test cutoff invariance, scope isolation, cache keys, label/manifest exclusion,
no hidden-edge access, exact posterior updates on tiny worlds, all-query accounting,
and replay equality. Check account renaming with exchangeable tie-breaking.
Preserve currency distinctions and transaction multiplicity. Confirm that one
acquired record enters the shared evidence once, while case loss is counted once
per distinct target. Report group-level results to expose duplicated-case gains.

Run label-shuffling, nuisance-only, no-overlap, and uninformative-sharing controls.
Any action-ranking advantage that depends on hidden identities, offline global
embeddings, or label-informed queue construction invalidates the result.

## 11. Stop/continue gates

| Gate | Required evidence | Spending/decision |
|---|---|---|
| Novelty mapping | Specific difference from CAMA, BRiG-AFA, PEGE, successor features and acquisition pruning; dense equation explicitly inherited | Up to $500 for small experiments; most work is reading/design |
| Data/interface feasibility | Useful historical prediction signal, natural shared evidence, and a defensible query-cost model | Cumulative cap $2,000; stop if the application is manufactured |
| Strong-baseline pilot | Headroom beyond shared-cache local, coverage, global greedy, and nonmyopic adaptation | Cumulative cap $4,000 before broad training |
| Method gate | Held-out benefit beyond best matched baselines; mechanism ablations support the explanation | Unlock main experiments, not automatically the entire budget |
| Generality gate | Independent financial generator and a meaningful additional setting; five-seed headline results | Unlock final confirmation/reproducibility work |
| Submission gate | A concrete original result, credible baselines, clean tests, limitations, and reproducible artifacts | Submit only when the result warrants the claim |

Stop or narrow if gains come only from absent caching, paying fewer duplicate
queries, high-degree heuristics, a weak risk model, artificial enriched queues,
or inconsistent clocks. A simple global planner winning is a finding, not a reason
to weaken the baseline. If the core method is already present in the literature,
do not spend $20,000 hoping scale will establish novelty.

## 12. $20,000 AWS/Azure research budget

These are maximum planning allocations, not quoted prices, required purchases,
or a promise of a specific number of GPU hours. One primary cloud should hold each
dataset and most jobs; use the other for available credits/capacity, not distributed
training across clouds. Do not assume credits can pay reviewers or unrelated APIs.

| Allocation | Cap |
|---|---:|
| Finite-world pilot and small feasibility runs | $500 |
| Data indexing, initial predictors, feasibility workloads | $1,500 |
| Baseline reproduction and adaptation | $2,000 |
| Main method training and controlled ablations | $5,000 |
| Independent-generator, size/overlap, and robustness evaluation | $3,500 |
| Optional hosted/open-model agent comparisons on a fixed subset | $1,500 |
| Storage, logs, egress, and final artifact reproduction | $1,000 |
| Uncommitted reserve | $5,000 |
| Total ceiling | $20,000 |

Start with CPU preprocessing and one modest GPU. AWS [G6](https://aws.amazon.com/ec2/instance-types/g6/)
is an example of a suitable GPU family to benchmark, not a reserved instance or
current regional price quote. Choose Azure capacity using actual subscription
quota and credit coverage. Large multi-GPU machines are justified only by a
measured throughput or memory requirement. No foundation-model pretraining.

Before each sweep, estimate jobs × measured hours/job × current instance rate,
plus CPU, storage, egress, and applicable overhead. Train budget-conditioned policies
across several budgets rather than independently retraining every budget point.
Record failed runs and total compute; use checkpointable jobs for interruptible
capacity and fixed configs for final comparisons.

Set per-job wall-clock limits, bounded concurrent workers, idle shutdown, and a
resource inventory with cost tags. Use budget alerts as monitoring, not hard caps:
[AWS documents notification delays](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html),
and [Azure budgets do not stop consumption](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/tutorial-acm-create-budgets).
Keep the reserve for uncertainty and final confirmation rather than automatically
expanding the model. No cloud resources were launched in this revision.

## 13. Delivery schedule

### First 50 focused hours: credible interview prototype

| Work | Hours | Output |
|---|---:|---|
| Freeze queue protocol and review closest formulations | 6 | Contract and novelty/baseline map |
| Tiny worlds, gateway/cache, reference checks | 10 | Correct shared-acquisition environment |
| IBM sample audit and target/history risk baseline | 10 | First application-feasibility decision |
| Cached local, coverage, and global-greedy policies | 8 | First meaningful acquisition comparison |
| Small PyG risk/action-value prototype if justified | 10 | Preliminary learned result, not a paper claim |
| Notebook traces, figures, rehearsal | 6 | Honest interview demonstration |
| Total | 50 | Prototype and documented outcomes |

At 30 hours, show the validated environment, data audit, and strongest completed
baselines. Do not imply the research method or independent-generator study is done.
The exact interview date is not known; this is a relative work allocation.

### Research program: approximately 20–24 weeks

- Weeks 1–2: pilot and data/interface gates.
- Weeks 3–5: baseline reproductions, exact-world mechanism, shared-cache controls.
- Weeks 6–9: adaptive influence candidate only if scalar/dense baseline headroom exists; coverage and ablations.
- Weeks 10–13: independent financial generator and chronological/group transfer.
- Weeks 14–17: nonfinancial setting, analytical work, runtime/row/bulk-query controls.
- Weeks 18–21: preregistered final runs, five seeds, uncertainty, independent reruns.
- Weeks 22–24: manuscript, artifact release, limitations, and submission decision.

This assumes sustained solo research time; the $20,000 budget does not substitute
for implementation and analysis time. The course deliverable is a completed,
well-explained subset on its actual deadline. Verify current solo policy and course
requirements; the supplied PDFs are from 2025. Prepare the blog and Colab throughout,
and obtain the required private-draft review before publishing the course blog.
Do not invent future conference deadlines; select the next eligible official call
after the experiments and manuscript are ready.

## 14. Paper and interview framing

**Working title:** Investigate Together: Joint Evidence Acquisition for Related
Fraud Alerts.

**Proposal abstract:** Fraud investigation often involves several alerts connected
through shared accounts. An evidence request can therefore affect multiple
decisions, while its value may become apparent only after follow-up queries. We
will study sequential acquisition over a partially observed transaction graph
under a queue-level budget. To separate planning benefits from caching, all methods
will share the same evidence store, permissions, and retrieval accounting. We will
test whether adaptive computation of action–case influence preserves the benefits
of joint acquisition at lower end-to-end computational cost. Standard scalar risk
regression and common-policy reward decomposition are baselines. Exact finite
worlds will diagnose decision quality; public synthetic financial datasets and a
nonfinancial relational task will test generalization. Comparisons will include
cached case-local policies, global acquisition heuristics, and nonmyopic planners.
The contribution will depend on demonstrated gains beyond these alternatives and
on evidence that the assumed retrieval constraints matter.

**Plain-language pitch:** “A bank has a queue of alerts, and several can involve
the same accounts. I am studying how to choose the next evidence request for the
whole queue. The key comparison gives every method the same cache, so I can test
whether joint planning actually improves detection rather than merely avoiding
duplicate downloads.”

Five central figures: one query informing several cases; the shared/delayed-value
mechanism; recall versus retrieval cost; independent-generator and size transfer;
and the cache/objective/planning ablations. Include an explicit failure case.

Before results, say “I designed” or “I am testing.” After results, insert the actual
held-out measurements, runtime, and limitations. Do not claim actual bank deployment,
causal proof of laundering, real dollar savings, or conference readiness from a
simulator demonstration.

**First implementation task:** the shared evidence gateway and tiny queue worlds,
alongside the existing IBM data audit. The previous scaffold stays archived; no
application code is restored by this plan.
