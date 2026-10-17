# Current research handoff — September 24, 2026

Upload this file to another chat. It contains the complete current plan below; no earlier conversation is required. The previous adversarial-distraction proposal was superseded after an external audit.

## Suggested message

> Use this plan as the current research direction. Help me sharpen the experimental contract and assess the remaining publication risk. The chosen question is whether shared-account reasoning reduces avoidable evidence-search failure under ordinary graph growth. Challenge the finite graph distributions, exact observation-constrained reference, equal-information baselines, and transfer assumptions. Do not substitute successful full-information traces for an information-constrained policy. Verify relevant literature before making novelty claims. Do not write implementation code unless I explicitly ask.

## User goals and constraints

The user wants a solo CS224W project with a credible path to a NeurIPS/ICLR/ICML main-track paper and relevance to Socratix AI's MTS role. The question and motivation must be exceptionally clear. The original planning window was an interview in 1–2 weeks, around 30–50 focused hours, public data only, and an initial $300 ceiling. Treat these as planning assumptions; no exact current interview date is supplied. The broader research horizon is the quarter plus 2–3 months.

The user wants to implement from scratch. An earlier generated scaffold was unfinished, checked only for compilation, and archived outside the active repository. No models, benchmark results, or validated environment tests exist. IBM HI-Small transaction and pattern files have been downloaded, but full ingestion and the historical-data audit are not complete.

The completed Socratix take-home was a role-aware conversational BI system. It enforced authorization in application code, bounded operations, validated result releases, scoped identity, and auditable evidence; it documented remaining cross-query inference risks. It was not a graph detector. Reuse lessons, not company fixtures.

## The decision that must remain explicit

The earlier plan asked whether distracting graph additions could cause a policy to miss still-reachable evidence. The external audit correctly observed that a successful path can exist even when the available observations do not identify it. The revised project measures achievable discovery under legal observations, then the policy's shortfall from that reference.

The optimum knows the declared generating distribution but not the realized hidden graph. That distributional privilege must be disclosed. It is a diagnostic, not a fair finite-data deployed competitor. The exact-reference equations and oracle supervision are not novel contributions by themselves.

The initial study has no strategic attacker and no learned classifier: a symbolic verifier scores a precise temporal graph witness. Learned graph and identity-aware set models are compared only after strong simple and planning baselines establish useful headroom. Public-data AML work follows a separate evidence-sufficiency gate.

The full plan includes primary references checked on September 24, 2026. This is a focused check, not exhaustive novelty clearance or replication. Current course rules, the exact financial workflow bottleneck, and dataset-specific timing claims remain to verify.

---

# Complete current plan

# When More Transactions Hide the Evidence

**Avoidable search failure in budgeted graph investigation**

Active plan, September 24, 2026. Incorporates the supplied external research audit
and replaces the adversarial-distraction plan. Implementation starts from scratch.
There are no trained models or experimental results. The direction is chosen;
the hypothesis and publication potential remain to be tested.

## 1. Final research question

> **Can reasoning about shared accounts reduce avoidable evidence-search failures
> as transaction graphs grow, beyond strong simple and nonmyopic policies at the
> same query budget?**

An investigator has time to request only a few account histories. More activity
creates more possible trails. Can it use connections in records it has already
retrieved to choose better next requests?

The scientific question is whether a policy misses evidence it had enough
information to find. Evidence existing somewhere in the database does not imply
that a limited investigator could reasonably locate it. We will measure that
distinction before claiming policy failure or developing a new method.

For example, returned records may connect two investigative branches to the same
account. That relationship may make a particular continuation worth checking,
although the confirming transfers remain hidden. Unrelated activity adds competing
requests. We will test whether policies use the visible relationship or spend
their budget elsewhere. The cue must be informative under an explicit distribution
and must not itself reveal the complete answer.

**Chosen direction:** analysis of relational evidence acquisition under ordinary,
non-adversarial graph growth. A small graph policy is an experimental instrument.
Method development follows only if the analysis identifies a consequential gap.

The old replay experiment establishes that a successful path exists. It does not
establish that the investigator could identify it from its observations. The new
reference must make decisions without seeing the realized hidden graph. This is
the main change in the plan.

## 2. Why this fits the user's goals

### Socratix and the take-home

Socratix's MTS posting describes gathering evidence across systems, tool use,
retrieval, evaluation, and auditability. Evaluating whether an investigator chooses
useful evidence fits those responsibilities. A query selector could later be
useful, but no actual product weakness or per-account retrieval bottleneck has
been established. If a permitted bulk query is cheap, this sequential interface
may be inappropriate for that workflow. [MTS role](https://www.ycombinator.com/companies/socratix-ai/jobs/JtqwNF7-member-of-technical-staff-ai-agents).

The completed take-home enforced authorization and validated result release for
conversational BI. This project asks what the agent should retrieve within that
boundary. Reuse scoped-operation and audit-trace principles; use public data rather
than company fixtures. The take-home was not a graph detector and explicitly
retained unresolved cross-query inference risks.

### Course and publication

Keep Project 1: a substantive PyG application/analysis, public-data experiments,
a readable blog, and Colab. The supplied 2025 materials allocate 50 final-submission
points to the blog, 25 to Colab, and 5 to the Medium account. Confirm current solo
eligibility and deadlines through course channels. The controlled pilot does not
replace the intended public-data course component. Follow the supplied private
blog-draft review requirement, subject to current course instructions.

The research target remains a NeurIPS/ICLR/ICML main-track paper if results justify
it. No exact submission date or acceptance probability is assumed. A useful course
project and interview demonstration are deliverables; publication is a conditional
outcome.

## 3. Candidate contribution and novelty boundary

The proposed contribution is an explanation of **when additional branches increase
the gap between achievable evidence discovery and implemented search**, and which
observable relationships reduce that gap.

A paper should establish a general mechanism, explain it through controlled
comparisons, and demonstrate a practical response on held-out structures and an
independent meaningful setting. The response need not be a new neural architecture.
If a simple rule solves the effect, report it; continue toward a paper only if a
substantive general analysis remains. Do not engineer harder examples solely to
make a GNN necessary.

### Closest work checked for this revision

| Work | Overlap | Consequence |
|---|---|---|
| [Selective network discovery, 2021](https://link.springer.com/article/10.1007/s41109-021-00365-8) | Learned budgeted discovery in partially observed networks | Learning which node to query is established |
| [Hierarchical costly-feature acquisition, 2024](https://link.springer.com/article/10.1007/s10994-024-06565-4) | Sequential acquisition from structured data | Dynamic structured retrieval is not the contribution |
| [ACO, ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html) | Nongreedy acquisition of jointly useful features | Compare beyond immediate-reward policies |
| [SEFA, ICML 2025](https://proceedings.mlr.press/v267/norcliffe25a.html) | Supervised acquisition over possible unobserved realizations | Supervision instead of RL is not novel |
| [PEGE, June 2026 version](https://arxiv.org/html/2601.16233v2) | Planning with distributions over graph expansions | Uncertain graph completion inside a policy is already studied |
| [AgentGL](https://arxiv.org/abs/2604.05846v2) | RL-trained graph-tool exploration; author record lists ACL 2026 | A graph agent alone is insufficient |
| [CDEG, August 2026 version](https://arxiv.org/abs/2608.22899v2) | Missing versus unused diagnostic evidence | Acquisition-versus-reasoning diagnosis is already explicit |

Primary pages were checked for this revision, including PEGE's observation
formulation. This is a focused positioning check, not exhaustive novelty clearance
or reproduction. PEGE reveals the induced graph on discovered vertices; our
gateway releases only queried records. That difference matters experimentally
but is not sufficient novelty by itself.

The strongest reviewer objection is that this is a standard partially observed
search problem with a planted cue and a model suited to the generator. Answer it
with informative/null controls, strong equal-information competitors, independent
generators, and a result explaining more than one handcrafted construction.

## 4. The measurement that changes the project

Let D be a declared distribution over complete graphs and B the query budget.
Let W(h) indicate whether released records in history h contain any valid witness.
A witness establishes a precisely defined graph property, not money laundering.

\[
V_\pi(D,B)=\Pr_{G\sim D,\pi}[W(h_B)=1],\qquad
V^*_{\mathrm{obs}}(D,B)=\max_{\pi\in\Pi_{\mathrm{legal}}}V_\pi(D,B).
\]

The reference knows D but acts only on legal observations. It cannot see the
realized hidden completion, witness, label, or successful reference trace.

\[
R_\pi(D,B)=V^*_{\mathrm{obs}}(D,B)-V_\pi(D,B),\qquad
\Delta R_\pi=R_\pi(D_{\mathrm{grown}},B)-R_\pi(D_{\mathrm{clean}},B).
\]

For a fixed policy, observed success loss equals:

\[
V_\pi(D_{\mathrm{clean}},B)-V_\pi(D_{\mathrm{grown}},B)
=\left[V^*_{\mathrm{obs}}(D_{\mathrm{clean}},B)-V^*_{\mathrm{obs}}(D_{\mathrm{grown}},B)\right]
+\Delta R_\pi.
\]

The first term is the change in the information-limited optimum; the second is
the change in policy shortfall. This is an accounting identity, not a new theorem.
Growth can sometimes expose useful information: report signed changes rather than
assuming the optimum must deteriorate. R includes estimation, representation, and
planning limitations; further controls are needed to attribute the cause.

**Sanity check:** with K exchangeable branches, one hidden witness, and failed
queries that only eliminate a branch, q distinct queries succeed with probability
min(1, q/K). Every realized graph still has a successful one-query sequence. Replay
would not establish that the remaining failure was avoidable.

Keep positive prevalence p fixed across growth conditions. Optimize the reference
over the full positive/negative mixture; never reveal or condition its actions on
the hidden label. Report V over all cases and positive-case discovery success.
Their regrets differ by the same factor p. A negative case cannot earn discovery
reward; failure to retrieve a witness is not a learned low-risk decision.

The reference has privileged knowledge of the true distribution. It is an
information/planning diagnostic, not a fair finite-data deployable competitor.
Later add planners with priors estimated from training data only.

## 5. The first experimental contract

### Observation gateway

- Account nodes and directed, timestamped transaction edges, preserving parallel
  transactions. Initial evidence is the target and its sender/receiver.
- The target sender is the root for the pilot graph property.
- Query the complete outgoing history of a discovered, unqueried account. Each
  request costs one call, including an empty response.
- Only returned records become visible. Knowing both endpoints does not reveal
  an unqueried edge between them.
- All supporting records precede the target timestamp and lie within the declared
  history window. No whole-graph degrees, learned account IDs, hidden neighbors,
  generator metadata, or labels enter policy state.
- Fixed horizon; terminate early only after verified discovery or exhaustion of
  legal actions. No learned stopping, pagination, outages, or attacker initially.
- Record calls, returned rows, and runtime separately. Unit calls are not verified
  bank API costs or analyst hours.

### Exact witness and budget

The witness consists of two simple directed paths of **exactly three edges each**
from root s to account z. They have distinct internal accounts and share only s
and z. Timestamps strictly increase along each path; all six records precede the
cutoff and must have been returned.

Example: s → a → c → z and s → b → d → z. Queries to s, a, b, c, and d can reveal
the witness in five calls. Score every valid witness, not just the planted one.
Assign positive/negative labels by checking the full graph.

Use **six calls as the primary budget**, with four and eight as diagnostics. This
changes the audit's proposed four-call condition: the chosen longer witness
requires at least five outgoing-history queries. Four calls become an impossibility
check. The extra depth allows intermediate evidence to guide later choices before
a witness is complete.

### Finite distributions and ordinary growth

Cap the pilot at 12 accounts and 64 hidden completions per declared distribution
used for exact evaluation. These are computational limits, not enumeration of
all graphs of that size. Include every hidden template in the declared support
consistent with initial information; never select a completion set using the test
graph's hidden template ID.

Generate paired conditions with 0, 2, and 4 added edges. Preserve original records,
the target, original timestamps, and the full-graph symbolic label. Added accounts
count against the size cap. Growth is sampled independently of the tested policy.

In the strict initial family, additions must create no new valid witness. This is
an evaluator-verified synthetic restriction, not a realism claim. Log rejection
rates and the resulting effective distribution. Responses are complete, so page
overflow cannot explain a failure. Replay original queries through the actual
grown environment as a reachability/accounting check.

Vary relational informativeness separately:

| Condition | Required property |
|---|---|
| Null | Shared-account observations do not predict useful continuations after nuisance matching |
| Weak signal | These observations weakly change continuation probabilities |
| Strong signal | These relationships give a stronger but imperfect continuation signal |

The cue must be a relationship in paid-for records, visible before completing a
witness. It cannot be a supplied “useful branch” feature. Match degree, row-count,
amount, and timestamp distributions where possible; measure residual shortcuts.

**The first four hours produce explicit completion tables and probability weights.**
Verify signal strength from conditional continuation values. The null cannot
merely be named “null.” These distributions are a design to implement and validate,
not an already working benchmark.

### Exact observation-constrained reference

For history h, retain all completions consistent with its query responses and
weight them by their posterior probabilities under D. Use the Bellman recursion:

\[
F(h,r)=\max_{a\in A(h)}\sum_o P_D(o\mid h,a)F(h\mathbin{\Vert}(a,o),r-1).
\]

Set F=1 after a witness is acquired; otherwise F=0 at zero budget or no legal
actions. Average over responses before future action maximization. Averaging
full-graph optimal traces would leak future information and is not this reference.

Memoize observation/belief states and check computation early. Reduce the finite
support or graph size if necessary, documenting the reduction. Never substitute
an approximate planner while calling it exact. Approximation is a later study.

## 6. Baselines and learning

### Before training

Compare random queries without replacement, breadth-first, recency, and a strong
path-completion/diversity heuristic. The latter receives visible partial paths,
shared endpoints, query completeness, and branch coverage—every relational fact
available to the GNN. Freeze its scoring rule and validation tuning before testing.

Add one-step and two-step planning under the declared prior and the full-horizon
exact reference. Short-horizon reward can be zero before a multi-query witness;
beating that alone is not compelling. Include a stronger two-step planner with a
fixed heuristic rollout for its remaining horizon. Disclose these planners' prior
knowledge separately from ordinary deployable baselines.

The first decision is whether meaningful opportunity remains after strong simple
and planning comparisons, not whether a neural policy can beat random search.

### Only if headroom exists

Train a budget-conditioned PyG action-value model on conditional values from
training distributions. Start with three message-passing layers of width 64,
directed/reverse relations, observed transaction features, root/target indicators,
queried/unqueried state, graph pooling, and a legal-action head. This is an
experimental instrument, not an asserted new algorithm.

Compare with an identity-aware set/attention model given identical records,
endpoint equality information, query state, budget, training cases, and supervision.
Match tuning effort and report parameter count and compute.

Use posterior expected action values, including ties and equally useful actions.
Do not label one full-information successful action as uniquely correct. Check
coverage of states visited by the learned policy and address gaps using training
worlds only; never supply test-state values to training.

Run graph versus set model, each with clean-only versus matched growth-augmented
training. This separates representation from exposure to larger graphs. A frozen
policy's growth loss is a different comparison from retrained-policy performance.

Tree-unrolling or removed identity links are information-removal ablations. They
cannot establish an equal-information architecture advantage.

Later add a structured hierarchical acquisition adaptation and a sampled-completion
nonmyopic planner informed by ACO/PEGE, under our gateway. Published results using
other tool access are not comparable. An LLM baseline and RL training are optional
later additions, not initial requirements.

## 7. Evaluation, correctness, and decision gates

Primary condition: six calls, four added edges, strong relational signal. Publish
the full budget × growth × signal grid. Enumerate expectations where feasible.
Use template/structure-disjoint training, validation, and test distributions and
paired clean/grown test worlds. Report uncertainty across independent templates
and learning seeds, not permutations of the same graph treated as new samples.

Report V, its exact reference, R, delta-R, positive-case discovery rates, calls,
rows, latency, witness counts, growth rejection, and replay eligibility. Use three
learning seeds for the pilot and five for headline research findings. Replay is
a reachability check, not the definition of regret.

Required tests:

- The reference reproduces the exchangeable-branch result and zero success below
  the minimum required query count.
- Identical released histories yield identical conditional values and deterministic
  policy scores regardless of the realized hidden completion.
- Hidden labels, templates, witnesses, growth manifests, and reference traces
  cannot enter deployable features.
- Future records cannot affect earlier cases; no induced edges are revealed;
  every query and replay obey discovery and cost accounting.
- An independent brute-force path enumeration checks the witness verifier and
  growth preserves labels and the initial family's witness set.
- Scores are equivariant to account renaming. Symmetric ties are randomized
  exchangeably; check distributional invariance or couple randomness under the
  renaming. Lexical-ID tie-breaking is not a valid invariance strategy.
- Learned policies use no test completion values or hidden test priors. Privileged
  exact-reference computations remain isolated from training.
- Keep currencies separate and parallel records intact; run signal-null and
  shuffled-supervision controls before substantive interpretation.

**Gate 0, about hour 16:** the environment and exact reference pass their validity
and sanity checks. No mechanism claim is made before this.

**Gate 1, about hour 22:** as a practical screen, seek at least five percentage
points of growth-induced excess shortfall for the strongest deployable simple
policy at the primary condition, on the positive-discovery scale. Inspect the
optimum and every control. This is an engineering threshold, not a significance
guarantee or publication bar. If the optimum explains the loss, or closure/diversity
essentially attains it, stop the proposed neural-method development.

**Gate 2, completed pilot:** require an improvement over the strongest matched
simple and learned competitors on held-out templates. A planning target is to
recover half of Gate 1's excess shortfall while losing no more than two percentage
points of clean discovery. Report intervals and tradeoffs. If augmentation helps
both architectures equally, narrow the mechanism claim accordingly.

**Gate 3, paper development:** the mechanism survives an independently designed
witness family, stronger planners, imperfectly estimated priors, larger structures,
and an independent meaningful application. Otherwise retain a rigorous course
project without calling it conference-ready.

## 8. Public financial-data application gate

IBM HI-Small remains the first intended public application; its raw files are
downloaded. Full ingestion and AML policy training are outside the initial pilot.

Before acquisition training:

1. Verify version, license, timestamps, pattern coverage, and the prior plan's
   reported September 1–10, 2022 primary period/all-positive later tail using the
   documentation and files. These claims were not revalidated in this revision.
2. Use `(bank_id, account_id)` identity, exact monetary evidence, distinct currencies,
   historical responses, and evaluator-only labels.
3. Audit annotated pattern visibility before each target. A laundering label may
   depend on future transactions. Start with a 48-hour window and warmup; tune
   alternative windows on validation only.
4. Compare target-only prediction with historical full-authorized-evidence models
   on chronological splits. A model score is a diagnostic reference, not an
   information-theoretic ceiling; failure also requires checking model adequacy.
5. Proceed if useful historical relational signal remains. Otherwise choose an
   explicit later review-time task or another public task. Do not silently admit
   future evidence to rescue the result.

For AML evaluation, use approximately 60/20/20 chronological target splits with
timestamp groups intact and training-only preprocessing. Preserve full historical
graphs while sampling targets. Initially include all eligible positives and a
uniform negative sample with inclusion probabilities for weighted estimates;
confirm headline findings on full eligible populations. Address dependence from
overlapping neighborhoods.

Report AP and recall at fixed review capacity versus retrieval cost. Do not claim
exact regret when the true financial distribution is unknown. Frozen-classifier
replay is a limited acquisition diagnostic, not causal identification.

LI-Small is another simulation from the same IBM generator. It is not independent
generator transfer. TransXion is a possible second generator after version and
label checks. A nonfinancial extension must have meaningful joint evidence or
witness semantics; an arbitrary ogbn-arxiv adaptation is not mandatory.

## 9. Schedule and scope

Planning constraints: one student, about 30–50 focused hours before the interview,
public data, and an initial $300 ceiling. No exact interview date is assumed.

| Work | Hours | Output |
|---|---:|---|
| Freeze witness, interface, distributions, comparisons | 4 | Contract and completion tables |
| Gateway, verifier, legal replay, exact reference | 12 | Validated finite-world experiment |
| Simple policies and information-null controls | 6 | Gate 1 result |
| GNN and equal-information learned comparator, if justified | 10 | First learned comparison |
| Held-out tests, seeds, failure analysis | 6 | Gate 2 evidence |
| Figures, trace walkthrough, documentation | 6 | Interview artifact and notebook |
| Total | 44 | Completed pilot, positive or negative |

With only 30 hours, prioritize the valid reference, baseline result, a modest
preliminary PyG comparison where justified, and a readable notebook. Do not promise
a completed AML model. A 50-hour allowance leaves six hours for contingency.

Use CPU computation initially and measure exact-state growth early. Try local
small-model training before renting a GPU. Retain up to $150 compute, $50 optional
later API comparisons, and $100 reserve. Spending is not required to begin.

Do not put a dashboard, agent framework, adversarial generator, stopping policy,
outage model, or broad data pipeline ahead of the pilot. Notebook graphs and
replayable step tables are enough for the initial demonstration.

### Through the quarter and following months

- Weeks 3–5: second independent witness family, larger held-out structures, both
  query directions, prior-estimation controls, and strong acquisition comparisons.
- Weeks 6–10: pass the financial-data gate, run public-data experiments, and complete
  the course blog/Colab. Revise the public task explicitly if the AML gate fails.
- Following 2–3 months: independent application, deeper analysis or a justified
  method, full evaluations, literature refresh, reproducible release, and an
  evidence-based main-track submission decision.

## 10. Framing and immediate next step

**Proposal abstract.** We will study whether transaction-graph growth causes
evidence-acquisition failures beyond those forced by limited observations. Policies
will discover verifiable relational witnesses through charged local history
queries. Finite graph distributions will permit exact comparison with an optimal
policy that knows the distribution but not the realized hidden graph. We will
vary branch growth and the informativeness of shared-account relationships to
separate changes in achievable discovery from policy shortfall. Strong path-based
heuristics, nonmyopic planners, and equal-information learned models will test
whether graph representations add value. Public synthetic financial data will
be introduced after verifying historical evidence sufficiency. Conclusions will
distinguish graph-property discovery from money-laundering classification and
condition further method development on held-out evidence of an avoidable gap.

**Interview explanation before results:** “My take-home focused on what an agent
is allowed to retrieve. This project asks whether it chooses useful evidence as
the number of possible trails grows. I'm designing small graph experiments where
I can measure what the available information permits, then test whether tracking
shared accounts improves the next query. That separates bad search choices from
evidence that was impossible to locate reliably.”

Five intended figures: query task; exchangeable-branch sanity check; discovery
versus growth with the exact reference; graph/set/heuristic comparison; successful
and failed traces. Generate results from saved manifests and distinguish small
demonstration runs from the complete experiments in the notebook and blog.

**Immediate next step:** write the tiny-world experimental contract and explicit
completion probability tables, then implement the gateway, symbolic verifier, and
exact observation-constrained planner. That is the first programming milestone.

This revision changes documentation only. The earlier scaffold remains archived;
implementation will begin from scratch. The original audit and older plans are
preserved in the repository's `docs/history/` directory.

