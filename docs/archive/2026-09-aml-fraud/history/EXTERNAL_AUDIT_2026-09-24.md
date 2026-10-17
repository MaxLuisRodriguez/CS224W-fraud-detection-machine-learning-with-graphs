## 1. Verdict: narrow the proposal, and change what the first experiment is meant to establish

**Retain budgeted graph investigation, but replace “Can investigators be distracted?” with a question about *avoidable* search failure under ordinary graph growth. Do not begin with an adversarial attack or a new reinforcement-learning algorithm.**

My strongest objection is not that your proposed phenomenon is implausible. It is that **the current experiment could succeed while establishing very little**: adding branches makes search harder, a successful reference sequence still exists, and a trained policy misses it. That does not establish that the policy had enough information to choose better.

Your paired replay is valuable. It establishes **retrievability under the modified environment**, provided discovery rules and costs are respected. It does **not** establish **identifiability from the investigator’s observations**. The handoff already recognizes that replay is not complete causal identification; this distinction needs to become the center of the research design. 

The literature also substantially narrows the available contribution. Learned selective network discovery predates current graph agents; hierarchical acquisition already handles dynamically revealed information; and PEGE explicitly plans over incrementally revealed graphs. CDEG additionally distinguishes missing evidence from evidence that was acquired but misused. Consequently, neither “learning to investigate” nor “separating acquisition from reasoning” is a defensible novelty claim by itself. ([Springer][1])

I would keep your three judgments separate:

| Judgment                                        | Assessment                                                                                                                                                                                                  |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Is the problem important?**                   | Yes, when evidence acquisition is genuinely constrained and missing a relationship matters. The particular per-account query interface remains a product assumption.                                        |
| **Is there an established scientific gap?**     | Not yet. There is a narrower, testable candidate: whether relational structure creates an exploitable separation between unavoidable information loss and avoidable acquisition failure under graph growth. |
| **Can you obtain convincing initial evidence?** | Yes, on tiny controlled graphs with an exact information-constrained reference policy. Not yet for real financial-crime investigations.                                                                     |

**Confidence:** high that the current broad novelty story is insufficient; moderate that the narrowed question warrants a tightly bounded pilot; insufficient evidence to call it a main-track contribution.

I would change this recommendation if the pilot shows that the optimal same-observation policy deteriorates just as much as ordinary policies, or that a simple graph-aware heuristic closes the entire gap. Those are reasons to stop the proposed method-development effort—not reasons to engineer harder distractors.

The file reports no trained models or experimental results, so this recommendation does not discard validated progress. 

---

## 2. Three serious candidate questions

### A. Current proposal: adversarial distraction of budgeted investigators

**Question:** Can bounded, evidence-preserving graph additions divert an investigator, and can training resist them?

The motivation is understandable, and a paired attack demonstration could be compelling. However, it combines several unresolved burdens: demonstrating avoidable failure, establishing a meaningful attacker, preserving the task’s semantics, and producing a defense beyond augmentation or adversarial training.

The closest overlaps include learned graph exploration, graph-retrieval poisoning, and learning-aware adversarial decision-making. LogicPoison attacks graph reasoning paths, while existing robust graph-exploration work already studies combining predictive exploration with protection against bad guidance. Their settings differ from yours, but they prevent broad claims that attacking or robustifying search is new. 

**Potential contribution:** a genuinely new mechanism or robustness principle for truthful, locally acquired evidence.

**Available evidence:** controlled graphs initially; synthetic financial stress tests later.

**Decisive test:** compare targeted and matched random additions against both strong simple policies and an optimal same-observation reference.

**Biggest risk:** an artificial attacker exploits a contrived policy weakness. **Not my recommendation for the starting project.**

### B. Narrower variant: avoidable search failure under irrelevant branch growth

**Question:** Does representing shared entities across branches help an investigator preserve evidence discovery as irrelevant branches accumulate, beyond strong nonmyopic and diversity baselines?

This retains your strongest assets: exact witnesses, bounded tools, replayable traces, and a substantive graph-learning component. It removes the need to justify an attacker before establishing the mechanism.

The overlap with hierarchical acquisition, active network discovery, and PEGE is substantial. The candidate contribution is therefore **an analysis of when graph growth causes excess policy failure, not a new formulation of graph acquisition**. ([Springer][2])

**Available evidence:** controlled graph distributions where hidden completions can be enumerated.

**Solo feasibility:** strongest of the three; the first result can be interpretable without training a large model.

**Decisive test:** measure growth-induced failure relative to an exact same-observation policy, with relational cues enabled and disabled.

**Biggest risk:** a cheap closure-aware diversity rule solves it, or the generator manufactures the entire advantage. **This is my recommendation.**

### C. Different problem: knowing when an incomplete investigation cannot support a decision

**Question:** Can a graph model maintain reliable abstention decisions when the policy collecting its evidence changes?

This targets a different output: not *which query next*, but *whether the acquired graph supports a sufficiently reliable conclusion*. The concrete concern is confusing “I found no suspicious relationship” with “I searched enough to justify that conclusion.”

It has a direct guardrail motivation, but generic graph uncertainty and “investigate before abstaining” are already occupied. ICML 2023 work constructs conformal prediction sets for GNNs; KnowGuard, ICLR 2026, combines evidence discovery with abstention in multi-round reasoning. ([Proceedings of Machine Learning Research][3])

**Potential contribution:** policy-shift-aware reliability under adaptively acquired graph evidence.

**Available evidence:** synthetic complete graphs with controlled acquisition policies; public synthetic AML labels for a limited application study.

**Decisive test:** calibrate under one acquisition policy, evaluate under another at matched budgets, and measure risk–coverage deterioration.

**Biggest risk:** an ordinary distribution-shift finding without a defensible correction or useful guarantee. Its strongest version needs more statistical development than the initial interview window permits. **Worth revisiting, but not the first commitment.**

---

## 3. The chosen research question

> **Does modeling cross-branch entity identity reduce avoidable witness-discovery loss under irrelevant branch growth, beyond strong diversity and nonmyopic acquisition baselines, at the same query budget?**

Here, a **witness** means a verifiable set of transaction edges establishing a precisely defined graph property—not proof of money laundering.

### Motivation in plain language

An investigator may need several individually inconclusive records to establish a relationship. Additional activity creates more possible trails. The important question is not whether more trails make searching harder. It is whether the investigator ignores relationships already visible in its evidence and therefore performs worse than its information actually permits.

### Concrete example

Suppose two partially explored trails may converge on the same downstream account. An already retrieved record creates a useful link between the trails, but the final confirming transfer remains hidden. The investigator can spend its remaining calls either completing that connection or opening unrelated branches.

If the observations favor completing the connection, missing it may be avoidable. If every remaining branch is observationally interchangeable, the same failure may be unavoidable. The experiment must distinguish those cases.

### Variables and falsification

**Independent variables:** irrelevant branch growth, availability of informative cross-branch relationships, query budget, and acquisition policy.

**Primary outcome:** the increase in discovery shortfall relative to the best policy with the same observation interface under the declared graph distribution.

**Hypothesis:** informative cross-branch relationships create exploitable acquisition opportunities that some learned policies lose under growth; a suitable graph representation can recover a meaningful portion beyond strong simple and nonmyopic baselines.

**Falsification:** the information-constrained optimum loses equally, simple baselines already attain it, or the apparent graph advantage disappears when competitors receive the same information and training data.

This question does **not** assume that a GNN will win.

---

## 4. Precise problem definition

### Environment, observations, and actions

Use a directed temporal transaction multigraph. An episode begins with a target transaction and its endpoints. The investigator sees only the target, previously returned records, discovered account identifiers, query-completeness state, and remaining budget.

The general action is a request for an incoming or outgoing history of a discovered account. Every response must come through the environment. **Do not automatically reveal all edges between discovered accounts:** knowing that two accounts exist is not equivalent to having retrieved their relationship.

Retain strictly historical evidence, separate currencies, parallel transactions, and identifiers used for equality/joining rather than learned account-specific embeddings. These are sensible interface choices from the proposal, not contributions. 

For the first experiment, simplify to **outgoing-history queries only**, complete responses, unit logical call cost, and a fixed horizon. Incoming queries, pagination, heterogeneous costs, and learned stopping should come later.

The terminal reward is one if the acquired records contain a valid witness and zero otherwise. A symbolic evaluator—not a neural classifier—determines witness discovery.

### Hidden information and learning

The hidden state includes unqueried transaction records and the complete graph. Policies cannot inspect witnesses, labels, generator metadata, or reference traces at deployment.

Training may use fully known synthetic graphs. All deployable learned competitors receive the same training cases and allowable observations.

The candidate GNN should encode the **observed graph plus query state**, then score legal actions conditional on remaining budget. Initially, train it using conditional action values from the small training environments. This is a convenient experimental instrument, not a claim of a new algorithm.

### The reference you currently lack

Let \(D\) denote a declared graph distribution and \(B\) the budget. Define

$$
V_\pi(D,B)=\Pr(\text{policy }\pi\text{ discovers a valid witness within }B),
$$

and

$$
V^*_{\mathrm{obs}}(D,B)
=
\max_{\pi\text{ using only legal observations}} V_\pi(D,B).
$$

Then define

$$
R_\pi(D,B)=V^*_{\mathrm{obs}}(D,B)-V_\pi(D,B).
$$

The primary diagnostic is

$$
\Delta R_\pi
=
R_\pi(D_{\mathrm{grown}},B)
-
R_\pi(D_{\mathrm{clean}},B).
$$

**A raw success decrease is not the same as increasing avoidable failure.** The latter requires positive excess shortfall after accounting for how much harder the observation problem became.

The optimal reference knows the declared generating distribution, **not the realized hidden graph**. Its advantage in distributional knowledge must be disclosed. It is an information/planning diagnostic, not a fair deployable competitor against a model that must estimate that distribution from finite data.

### Why feasibility is insufficient

Consider \(K\) exchangeable branches, exactly one containing a witness discoverable with one query. Suppose unsuccessful queries reveal nothing about the remaining branches except that the queried branch was wrong.

With \(q\leq K\) queries, the best success probability is \(q/K\). Yet every realized graph has a one-query successful reference sequence.

Thus, a perfect replay certificate can coexist with an unavoidable large failure rate. This elementary example is a sanity check—not a proposed theorem.

### What makes the graph matter?

The relevant property is **relationships across branches**, such as shared entities or partial path closure—not merely storing a list of observations in PyG.

An identity-aware set/attention model should receive the same returned records and equality information. A tree-unrolled model that duplicates shared accounts and removes their identity connections is an **information-removal ablation**, not an equal-information architecture comparison.

Necessary assumptions are that acquisition is meaningfully constrained, relational observations can sometimes predict useful continuations, and witnesses have unambiguous semantics. Unit costs, tiny graphs, complete source availability, and fixed horizons are initial conveniences.

There is no attacker in the first study.

---

## 5. Verified novelty matrix and corrections to the inherited literature

The matrix below reflects inspected full-text formulations and methods. Acceptance status is distinguished from an author/arXiv report where appropriate. Published performance claims have not been independently reproduced.

| Closest work                                                                                                                  | Setting, information access, and objective                                                                                                                                                      | Method and evaluation                                                                                                                                          | Implication for your contribution                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Classification with costly features in hierarchical deep sets — Machine Learning, 2024**                                    | Hierarchical observations; acquisitions can expose previously unavailable features/actions. Optimizes classification against acquisition cost.                                                  | Hierarchical Deep Sets and actor–critic acquisition; structured datasets, including a security-related application.                                            | Dynamic actions, structured retrieval, query costs, and stopping are already covered. Arbitrary shared-entity relationships and growth-adjusted witness discovery are distinctions to test, not automatic novelty. ([Springer][2])                                                    |
| **Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition — ICML 2024**                                       | Partially observed feature vectors with a known feature universe; seeks jointly useful acquisitions rather than only immediate gains.                                                           | A nonparametric acquisition-conditioned oracle, with policy learning/distillation; classification–cost evaluation.                                             | Nonmyopic acquisition and oracle-supervised query selection are not new. A graph adaptation must outperform an appropriate nongreedy competitor rather than merely a one-step policy. ([Proceedings of Machine Learning Research][4])                                                 |
| **Selective network discovery via deep reinforcement learning on embedded spaces — Applied Network Science, 2021**            | Partially observed networks; frontier queries reveal structure and node labels. Objective is target-node discovery under a query budget.                                                        | Network Actor Critic, task-oriented embeddings, and deep RL; synthetic and real-network studies.                                                               | This is a direct predecessor to “learning to investigate.” Your terminal multi-edge witness objective differs from accumulating revealed target labels, but the general acquisition problem is established. ([Springer][1])                                                           |
| **Adaptive Frontier Exploration on Graphs with Applications to Network-Based Disease Testing — NeurIPS 2025**                 | Known graph topology, unknown correlated node labels, frontier-constrained actions, discounted label-dependent reward.                                                                          | Gittins-index policy; optimality on forests in its stated setting; experiments also examine finite budgets and non-trees.                                      | Frontier constraints, graph-dependent information, and principled nonmyopic policies already have serious treatment. Its guarantee does not transfer to hidden topology and joint witness rewards. ([arXiv][5])                                                                       |
| **Policy-Embedded Graph Expansion — 2026 preprint, revised June 17**                                                          | Incrementally revealed graphs; its SAPOG formulation includes topology discovery and state-dependent rewards. Observations include the induced graph on discovered vertices and queried labels. | Samples possible graph expansions and incorporates them into planning; network-testing evaluation.                                                             | This is stronger overlap than the handoff emphasizes. Uncertain graph completion plus sequential acquisition is not an open formulation. Your narrower local-record interface and witness objective need explicit comparison. ([arXiv][6])                                            |
| **AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning — ACL 2026 according to author/arXiv records** | Tool-based exploration of text-attributed graphs with bounded search; tools include local and broader retrieval operations.                                                                     | RL, graph-conditioned training, and constrained search; node-classification/link-prediction benchmarks.                                                        | A learned graph-tool policy is not novel. Its tools are not equivalent to charged account histories, so an adapted comparison must disclose altered access and training. ([arXiv][7])                                                                                                 |
| **CDEG: Learning Decision-Critical Evidence for Long-Horizon Diagnostic Agents — August 2026 preprint**                       | Multi-round diagnostic evidence acquisition; separates missing, ignored, and blocking evidence.                                                                                                 | Contrasts successful/failed same-case trajectories, applies evidence interventions, builds an evidence–action graph, and guides later acquisition/reappraisal. | Paired trajectories and acquisition-versus-utilization diagnosis are already explicit. Its completion intervention inserts evidence into an evidence set; your actual, charged environment replay is different, but that distinction alone is unlikely to carry a paper. ([arXiv][8]) |
| **Robustification of Online Graph Exploration Methods — AAAI 2022**                                                           | Initially unknown weighted graphs; movement reveals local structure; objective is visiting all vertices, not finding one witness.                                                               | Combines predictive exploration with a robust fallback; theoretical guarantees and graph-exploration experiments.                                              | “Trust a learned explorer but fall back safely” is established. Different feedback and cost models prevent direct transfer of its guarantees. ([arXiv][9])                                                                                                                            |

### Important corrections and qualifications

**Two additional acquisition papers constrain the method story.** SEFA, *Stochastic Encodings for Active Feature Acquisition*, is ICML 2025 and reasons across possible unobserved realizations using supervised latent-variable learning. Its proceedings description alone is sufficient to rule out “supervised nonmyopic acquisition” as your contribution. BRiG-AFA, an August 2026 preprint whose full method I inspected, fits budget-specific Bellman risk-to-go predictors with a frozen classifier. Therefore, adding fitted-Q or backward value regression would not establish novelty either. ([Proceedings of Machine Learning Research][10])

**TransXion is not merely an unplaced preprint.** Its current record lists KDD 2026. However, the versions need to be kept separate: the first version and author repository explicitly describe adversarial graph edits, while the revised paper changes the framing substantially. Do not combine claims from different versions as though they describe one frozen benchmark. More importantly, anomaly synthesis or detector evasion is not the same problem as diverting a budgeted evidence-acquisition policy. ([arXiv][11])

**LogicPoison and KEPo are real overlaps, but not exact matches.** LogicPoison has an official ACL 2026 proceedings entry; its logical attacks modify knowledge/reasoning connections. KEPo’s paper identifies WWW 2026 and injects poisoned knowledge evolution. Neither should be described as a demonstrated attack on truthful, label-preserving financial histories under your exact query interface. ([ACL Anthology][12])

**The older leads need narrower descriptions.** *Adversarial Active Learning* is an AISec 2014 problem-setting/testbed paper, not evidence that your exact defense problem was already solved. *Deception Game* is CoRL 2023 and concerns learning-aware robot safety. The DFKI relational-acquisition lead is CODS-COMAD 2023 and involves acquiring relational information through human interaction; I verified its institutional abstract and metadata, not a full-text equivalence to per-case investigation. These are intellectual antecedents, not interchangeable experimental baselines. 

**IBM Multi-GNN is an implementation resource, not acquisition novelty.** Its repository identifies the AAAI 2024 directed-multigraph paper and NeurIPS 2023 synthetic-financial-data paper and exposes several graph-model adaptations. It does not supply a validated investigation policy or validate your historical-evidence assumptions. ([GitHub][13])

### What remains potentially new?

A narrow candidate contribution is:

> A demonstrated mechanism by which irrelevant graph growth increases acquisition-policy regret despite informative, legally observable cross-branch evidence—and an explanation of when relational learning fixes that mechanism beyond existing acquisition strategies.

**Verified distinctions:** the inspected works use different combinations of observation access, objectives, interventions, and evaluation.

**Unverified gap:** that this mechanism is sufficiently distinct, general, and unresolved to support a paper. A literature search cannot turn absence of an identical experiment into proof of novelty.

The strongest reviewer objection would be:

> “This is a small partially observable search problem with a planted relational cue. The oracle analysis is standard, and the GNN is an inductive bias for the generator.”

The research must answer that objection experimentally and analytically.

---

## 6. Contribution plan: analysis first, method only if demanded by the results

**Choose analysis as the primary contribution type.**

The intended contribution should explain a consequential distinction between:

* evidence that exists;
* evidence discoverable by an informed but non-clairvoyant investigator;
* evidence missed because of the implemented policy.

The equations above merely formalize that distinction. They are not themselves a new result.

A credible main-track development would need a substantive general finding: for example, a reproducible structural regime in which branch growth causes excess acquisition failure, a clear explanation of which observable dependencies matter, and a practical approach that addresses it across independently constructed environments. Any theorem would have to be developed and proved under explicit assumptions—not promised now.

It becomes merely incremental if the final result is “GNN plus augmentation beats clean GNN,” if one handcrafted feature explains all improvement, or if the only successful test is a generator designed around the model’s preferred representation.

Keep a method contribution optional. If a simple conditional planner works, that may indicate the useful result is an analysis of why existing policies fail. If everything strong works, there may be no research problem left.

Do not require four datasets, an LLM framework, a financial attack simulator, a new benchmark, and a theoretical guarantee before obtaining the first interpretable result. Your handoff correctly flags this scope problem. 

---

## 7. The smallest decisive experiment

### Experiment: growth-induced acquisition regret on an exactly solvable graph family

**The first deliverable should be a result comparing search policies with an exact same-observation reference—not an attack visualization.**

#### Data and task

Start with a finite family of tiny directed graphs, capped at approximately **10 accounts and 64 possible hidden completions per observational template**. These are proposed computational limits, not claims that every possible graph of that size must be enumerated.

Use a **temporal diamond** as the first witness: two distinct, temporally valid paths from a target endpoint to the same downstream account. Specify the exact permitted path lengths and internal-node constraints before generation. The target transaction may be visible at the cutoff; all supporting history must precede it.

Construct matched negative cases by rewiring so no valid witness exists. Check the entire graph, not merely the originally planted pattern.

Use outgoing-history queries, no pagination, unit costs, and **budget four as the primary condition**, with budgets two and six as diagnostics. This deliberately smaller setting replaces the original eight-call illustration for the pilot.

Generate paired clean/grown instances with **zero, two, or four added edges**, staying within the size cap. Additions must preserve the target, original records, positive/negative symbolic label, and a legal reference sequence where one existed.

The key additional factor is **relational informativeness**:

* In the null condition, visible branch relationships do not predict which continuation completes a witness.
* In informative conditions, already retrieved cross-branch identities change the distribution over useful continuations.
* A cue must become available through charged queries and must not itself constitute the completed witness.

Declare that dependence in the generator. Do not hide it as “realistic financial behavior.” Its realism is a later question.

#### First comparison: before training anything

Compute the exact expected success of the optimal policy by maintaining the distribution over completions consistent with each legal query history.

Compare it with uniform random search without replacement, breadth-first search, recency, and a **strong graph-aware heuristic** combining branch coverage with observed path-completion opportunities.

Also include one-step and two-step lookahead using the same declared completion distribution. These diagnose whether any gap reflects longer-horizon planning rather than a need for learned graph representation.

The purpose is to locate an interesting regime. If the exact planner and strong simple strategies behave similarly, there is no justification yet for training a “robust” GNN.

#### Second comparison: only if the first reveals headroom

Train a small budget-conditioned graph action-value model. Compare it with an identity-aware set/attention model and a structured acquisition baseline, using identical training cases, observation access, and supervision.

For the first learning test, use conditional action values computed over training completions—not a single omniscient successful trace.

**Why this matters:** two hidden graphs can produce the same observed history but have different successful reference actions. Treating each hidden graph’s successful action as the uniquely correct label gives conflicting, clairvoyant supervision. Also, one reference trace can wrongly penalize another equally successful route.

Compare clean-only training with matched growth augmentation. Do not introduce adversarial training until ordinary growth produces a meaningful, unexplained failure.

#### Controls that determine whether the result is believable

The essential controls are identifier-renaming invariance, no hidden-edge access, no witness or manifest leakage, no pagination, timestamp validity, and exact query accounting. Use all valid witnesses when scoring discovery, not only the planted one.

Match nuisance attributes and inspect degree, record-count, timestamp, and amount shortcuts. Structural rewiring that changes identity information is an intentional information intervention; report it separately from equal-information architecture comparisons.

Preserve actual modified-environment replay, but report eligibility and exclusions across all cases. For this first test, symbolic witness verification removes the final classifier as a confound entirely. The handoff’s frozen-classifier replay becomes useful later for AML labels. 

#### Metrics and decision criteria

Report discovery success, optimal-reference success, \(R_\pi\), and growth-induced \(\Delta R_\pi\) at the prespecified budget. Show calls and returned records separately.

For the finite pilot, enumerate completion probabilities where practical. For learned-policy evaluation, use independently generated training/validation/test templates, paired clean/grown cases, and uncertainty across independent templates and training seeds—not repeated identifier permutations treated as independent evidence.

**Continue only if** informative conditions retain meaningful optimal-reference headroom, strong simple policies leave part of it unused, and learning recovers some of that gap on held-out structure without a substantial clean-case penalty.

**Abandon the preferred framing if**:

1. The observed decline is explained by the same-observation optimum declining.
2. A simple closure/diversity rule consistently reaches the useful ceiling.
3. The advantage requires withholding identity information from competitors, relying on shortcuts, or training and testing on essentially the same graph templates.

Numerical tolerances can be fixed as engineering decision rules—for example, treating a gap of only a few percentage points as insufficient reason to build a new method—but they are not publication thresholds.

This experiment can reject the idea even if every paired replay succeeds. That is precisely why it is more decisive than the current proposal.

---

## 8. Two execution horizons

### Initial 30–50 focused hours

I would replace the original schedule’s eight-hour combined GNN/public-data block with a more conservative plan. The original schedule attempts environment validation, learned acquisition, and an AML diagnostic before the core information question has been answered. 

| Work                                                                 | Planning allocation | Required output                                   |
| -------------------------------------------------------------------- | ------------------: | ------------------------------------------------- |
| Freeze witness semantics, observation interface, and comparisons     |             4 hours | A short experimental contract                     |
| Build tiny environments, verifier, legal replay, and exact reference |            12 hours | Auditable search episodes and correctness tests   |
| Run simple policies and information-null controls                    |             6 hours | First stop/continue decision                      |
| Add small GNN and equal-information learned comparator, if justified |            10 hours | Initial graph-learning result                     |
| Held-out tests, additional seeds, and failure inspection             |             6 hours | Evidence supporting or rejecting the mechanism    |
| Figures, trace walkthrough, and documentation                        |             6 hours | Interview/course starting artifact                |
| **Total**                                                            |        **44 hours** | A completed pilot, not a promised positive result |

These are task-allocation assumptions, not delivery guarantees. At approximately the first half of that allocation, there should be enough evidence to decide whether a learned policy is warranted.

CPU computation should suffice for the tightly capped exact environments; check state-space growth early. Small-model training can use available hardware before spending the $300 budget. No LLM API is necessary.

Omit IBM ingestion, adaptive attacks, stopping, outages, a dashboard, and a second application domain from the initial critical path.

### Quarter plus another two to three months

The next stage is conditional on the pilot:

**Early quarter:** extend to a second independently designed witness family, allow both query directions, and reproduce the strongest acquisition comparisons under matched access.

**Later quarter:** test larger held-out structures and imperfectly estimated graph priors. Determine whether the proposed mechanism survives outside tiny enumerated worlds.

**Application gate:** before AML policy training, test whether historical evidence within the proposed window supports meaningful prediction at all. Use target-only and historical-full-evidence diagnostics, audit label timing, and measure how much positive-pattern information is visible before the target. A classifier’s full-evidence score is not an information-theoretic upper bound.

**Following months:** add one genuinely independent evaluation setting, deepen the mechanism or method, refresh the literature, and decide whether the evidence warrants a main-track manuscript.

HI-Small and LI-Small are not independent evidence of transfer across financial generators merely because they are different files. The plan itself treats them as simulations from the same generator. 

---

## 9. Revised framing package

### Working title

**Available Is Not Necessarily Findable: Avoidable Search Failure in Budgeted Graph Investigation**

### Proposal abstract

We will study when irrelevant graph growth causes avoidable failures in budgeted evidence acquisition. Using partially observed directed transaction graphs, we will require policies to discover verifiable relational witnesses through charged local queries. Tiny graph families will permit exact comparison with an optimal policy that has the same observations but knows the generating distribution. This reference will separate information dilution from failures of learned search. We will test whether shared-entity representations improve robustness beyond diversity heuristics and nonmyopic acquisition, using matched information and training data. Controls will remove relational signal, preserve labels and legal retrieval paths, and exclude pagination effects. Public synthetic financial data will be introduced only after the mechanism survives these tests. Any conclusions will distinguish graph-property discovery from financial-crime detection.

### Thirty-second interview explanation

> “My take-home focused on what an agent is allowed to retrieve. This project asks whether it chooses useful evidence once retrieval is permitted. I’m starting with small transaction graphs where I can distinguish three things: a successful path exists, the available information supports finding it, and the policy actually finds it. Then I’ll test whether tracking shared entities across branches improves search beyond simpler strategies. The initial goal is a trustworthy evaluation, not an unsupported claim of better fraud detection.”

The take-home bridge is supported by the handoff, but the new experiment remains proposed. 

### Three claims you can honestly make now

1. You have identified a precise distinction between budget-feasible evidence and evidence a non-clairvoyant investigator can reliably find.
2. You have designed a falsifiable evaluation using controlled witnesses, charged local queries, and an information-constrained reference.
3. You are testing whether relational learning is useful—not assuming that adding a GNN constitutes novelty.

Only later evidence would justify claiming that a mechanism occurs reliably, a method improves acquisition, the result transfers to synthetic AML classification, or the approach helps an actual financial investigation workflow.

Socratix’s public materials support the relevance of evidence gathering, retrieval, evaluations, and auditability. They do **not** verify that per-account query allocation is a customer bottleneck or that its agents have this failure. ([Socratix AI][14])

---

## 10. Remaining uncertainties and how to resolve them

| Uncertainty                                                               | Actionable resolution                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Is there a remaining novelty gap?**                                     | Use the pilot’s actual mechanism—not its title—to recheck PEGE, hierarchical acquisition, CDEG, and nonmyopic acquisition. The present audit establishes overlap and distinctions, not exhaustive novelty clearance.                           |
| **Does the query model reflect a real bottleneck?**                       | Establish whether production evidence tools impose meaningful incremental cost, latency, scope, or analyst-attention constraints. A cheap bulk join could make this particular sequential interface artificial.                                |
| **Do financial histories contain the needed evidence before the target?** | Audit historical visibility and run the target-only/full-history diagnostics before spending time on learned acquisition.                                                                                                                      |
| **Are dataset/version assumptions sound?**                                | Pin TransXion paper/code/data versions. Separately verify the IBM license, date range, claimed all-positive tail, and pattern-file coverage against documentation and the actual files; this audit did not inspect those large local datasets. |
| **Is a graph-learning advantage substantive?**                            | Test identity-aware nongraph competitors, explicit path-completion heuristics, and matched-data nonmyopic policies. Accept a simple-method win.                                                                                                |
| **What are the current course requirements?**                             | Confirm solo eligibility and deliverables through current course channels. The handoff contains summaries of 2025 materials, not verified 2026 permission or requirements.                                                                     |

**Recommended next step: freeze the tiny-graph experimental contract—witness definition, legal observations, graph-completion prior, growth conditions, and exact-reference comparison—then implement that experiment before any robust-policy training.**

[1]: https://link.springer.com/article/10.1007/s41109-021-00365-8 "https://link.springer.com/article/10.1007/s41109-021-00365-8"
[2]: https://link.springer.com/article/10.1007/s10994-024-06565-4 "Classification with costly features in hierarchical deep sets | Machine Learning | Springer Nature Link"
[3]: https://proceedings.mlr.press/v202/h-zargarbashi23a.html "https://proceedings.mlr.press/v202/h-zargarbashi23a.html"
[4]: https://proceedings.mlr.press/v235/valancius24a.html "Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition"
[5]: https://arxiv.org/html/2505.21671v3 "https://arxiv.org/html/2505.21671v3"
[6]: https://arxiv.org/abs/2601.16233 "https://arxiv.org/abs/2601.16233"
[7]: https://arxiv.org/abs/2604.05846 "[2604.05846] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning"
[8]: https://arxiv.org/abs/2608.22899 "https://arxiv.org/abs/2608.22899"
[9]: https://arxiv.org/html/2112.05422v1 "https://arxiv.org/html/2112.05422v1"
[10]: https://proceedings.mlr.press/v267/norcliffe25a.html "https://proceedings.mlr.press/v267/norcliffe25a.html"
[11]: https://arxiv.org/abs/2604.17420 "[2604.17420] TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering"
[12]: https://aclanthology.org/2026.acl-long.252/ "LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation - ACL Anthology"
[13]: https://github.com/IBM/Multi-GNN "https://github.com/IBM/Multi-GNN"
[14]: https://getsocratix.ai/ "https://getsocratix.ai/"
