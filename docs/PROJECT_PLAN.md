# Worth the Lookup

**Learning what to research before the call — budgeted relational retrieval for call-list prioritization**

CS224W (Fall 2026) · Project type 1: real-world application of GNNs · Plan dated 2026-10-10 (supersedes the September AML plans in [docs/archive](archive/2026-09-aml-fraud/ARCHIVE-NOTE.md))

---

## TL;DR

An outbound rep's call list is a graph problem. Prospects share accounts, events, colleagues, and activity, so one lookup ("enrich this account", "pull this contact's history") can sharpen the prediction for several prospects at once. Lookups are never free: enrichment credits cost money, LLM research agents cost tokens, and every second spent researching a prospect who never picks up is wasted.

**Question.** Given a call list of *N* prospects and a budget of *B* lookups, which lookups should a research agent make so that the resulting graph model ranks the list best?

**What we build.**

1. A relational GNN (PyG + RelBench) that stays accurate and calibrated when it has only seen *part* of each prospect's neighborhood.
2. Acquisition policies that choose lookups for the whole list, from simple heuristics up to a learned, batch-aware value model that credits a lookup for *every* prospect it informs.
3. Retrieval-budget frontiers ("how much accuracy per lookup") across several public relational databases.

**Data.** [RelBench](https://relbench.stanford.edu/) databases, which download automatically. Nothing to scrape or label.

---

## 1. Audit of the September plan

| September plan | Verdict | Change |
|---|---|---|
| Joint evidence acquisition across related alerts under a shared budget | **Keep.** This is the reusable research idea. | Same idea; alerts become prospects and bank accounts become company accounts and events. |
| IBM AMLworld as the only dataset, plus a hand-built investigation gateway | **Drop.** Synthetic, one generator, a custom environment to build and validate, and no tie to Nooks. | RelBench: several real relational databases with temporal splits and published GNN baselines. The "environment" becomes masking on PyG subgraphs. |
| Hand-built finite tiny worlds with dynamic programming as the first experiment | **Drop.** A week of work before any real data. | `rel-f1` (tiny, minutes to train) is the debugging world. |
| Dense continuation-value heads and adaptive action–case routing as the method | **Defer.** Heavy, and the audit already showed overlap with BRiG-AFA and successor features. | Start with mask-aware prediction plus a one-step batch-aware value model. Nonmyopic planning becomes a later extension. |
| 20–24 weeks, $20k, 16 baselines before any result | **Too big for a course project and too slow to show progress.** | Preliminary results this weekend. A 10-week course arc. A paper decision gated on measured headroom (§12). |
| Socratix framing | **Replace.** | Nooks framing (§2). |
| Honest claims, strict temporal cutoffs, equal-information baselines, shared cache for every policy | **Keep, non-negotiable.** | Unchanged. |

---

## 2. The problem in Nooks terms

Nooks sells an **AI Dialer**, **AI Sequencing**, **Signals and Intelligence**, and **AI Coaching** to sales teams ([nooks.ai](https://www.nooks.ai/)). Three facts about outbound calling define the problem:

- **Most dials never connect.** On a parallel dialer the rep doesn't know who will pick up, yet the brief must be on screen the moment someone does.
- **Research is a metered resource.** Third-party enrichment is billed per lookup. AI research agents bill per token and add latency. Rep attention is the scarcest of all.
- **Prospects are not independent.** Contacts share an account. Accounts share a parent company, a tech stack, an event, or a news story. One lookup is often relevant to several dials.

So the decision a research agent faces, before the session starts, is not "write a great brief for prospect *i*". It is **"which lookups, across the whole list, change who I should call first and what I should know?"** A pre-call brief agent such as SCOUT makes that choice implicitly for one prospect at a time. This project makes it explicit, batch-level, and learnable.

What the work offers Nooks:

- **Signals and Intelligence:** decide which signals are worth fetching and for which accounts.
- **AI Dialer:** rank the call list so researched, high-propensity prospects come first.
- **AI research agents:** a principled tool-call budget across many prospects, not per prospect.

---

## 3. Graph ML formulation

**Graph.** RelBench turns a relational database into a heterogeneous temporal graph: one node per row, one edge per primary-key/foreign-key link, and a timestamp per row (Relational Deep Learning; Fey et al., 2024).

**Call list.** At prediction time *t*, a batch $Q=\{q_1,\dots,q_N\}$ of target entities (e.g., users/prospects) with labels $y_i$ defined over $(t, t+\Delta]$. RelBench's task tables provide *t*, Δ, and leakage-safe labels.

**The lookable world.** $S(Q,t)$ is the *L*-hop temporal neighborhood that a standard RDL model would sample, with fanout caps that play the role of API page sizes, and only rows with time ≤ *t*.

**Actions.** Each action costs 1 by default.

- `list(q, R)`: fetch prospect *q*'s most recent ≤ *k* rows in child table *R* (e.g., "pull this contact's activity history"). This reveals those rows' features and their foreign keys.
- `get(u)`: fetch the row of an entity *u* that a revealed row references (e.g., "enrich this account / event / colleague"). Until it is fetched, *u* is a **stub**: we know it exists and who references it, but not what it is.

**Shared cache.** Anything fetched is visible to every prospect. A stub referenced by many prospects' rows is the "shared account": one `get` informs all of them. Every policy uses the same cache and budget.

**Predictor.** $f_\theta(Q \mid O)$ scores all prospects given the revealed set *O*. Stubs enter as a learned per-table `[MASK]` embedding.

**Objective.** Choose a policy π to maximize $\mathbb{E}\big[M\big(f_\theta(Q \mid O^{\pi}_B), y\big)\big]$ subject to total cost ≤ *B*, with *M* ∈ {AUROC, AP, Precision@K}. *K* = ⌈0.1·*N*⌉ is "the calls a rep actually gets to make first".

**Why this isn't trivial.**

- The value of `get(u)` is summed over every prospect *u* touches, so per-prospect reasoning undervalues shared entities.
- The ranking metric couples prospects even when they share nothing: a lookup about prospect *i* matters only if it can move *i* across the top-*K* cutoff, which depends on everyone else's scores.
- `get` depends on an earlier `list` (two-step value).
- A model trained on full neighborhoods has never seen missing ones.

---

## 4. Data: public and auto-downloading (RelBench)

`pip install "relbench[example]==3.0.2"`, then `relbench.load_dataset("rel-avito").load_task("user-clicks")`. RelBench 3.x downloads from the Hugging Face Hub; the 2.x `get_dataset`/`get_task` API no longer exists. Each dataset ships the database, temporal train/val/test splits, task tables, and maintained GNN and LightGBM baselines. Nothing to collect.

**Headroom rule.** Lookups can only help if relational context helps at all, so the primary tasks are those where the RelBench GNN clearly beats LightGBM on the target row alone (current v3 leaderboard, test AUROC):

| Role | Dataset / task | GNN vs. LightGBM | Sales analogue |
|---|---|---|---|
| Debug world | `rel-f1` / `driver-top3` | 0.680 vs. 0.649 | — (tiny: 857 drivers, ~1.4k training rows; validates the pipeline in minutes) |
| **Primary: sparse sharing** | `rel-avito` / `user-clicks` (also `user-visits`) | 0.666 vs. 0.548 (0.603 vs. 0.532) | **Intent signals.** Users ≈ prospects; ads, categories, and locations they search and visit ≈ shared accounts and products; `PhoneRequestsStream` is literally "asked for the seller's number". Will this prospect engage in the next 4 days? |
| **Primary: dense sharing** | `rel-event` / `user-ignore` | 0.804 vs. 0.773 | Will a prospect ignore ≥3 invitations next week? Invitations ≈ touches; events ≈ campaigns; friendships ≈ buying-committee links. Modest headroom, but the most shared entities. |
| Large headroom | `rel-stack` / `user-engagement` | 0.906 vs. 0.627 | Will a user contribute next quarter (product-led expansion signal)? 1 GB download; most of the gain comes from the user's own activity, which tests `list` vs. `get` sharply. |
| Excluded | `rel-event` / `user-repeat` | 0.686 vs. 0.689 | No graph advantage on the v3 leaderboard, so no room for lookups to help. |

Sizes:

- `rel-avito`: 450 MB download; 98k users and 6M ads.
- `rel-event`: 119 MB; 38k users, 3.1M events, 30M friendship rows.
- `rel-f1`: 1.4 MB.

Development uses the **validation** split. RelBench masks test labels by default in `task.get_table("test")` as a leakage guard; the test split is evaluated once, at the end.

**Measured sharing ([P4](modules/P4-why-these-datasets.md); measured 2026-10-16, reproduced 2026-10-17).** Across 200 random call lists of 128 validation-day users:

- On `rel-event`, **34%** of touched events are shared by 2+ prospects, and a shared cache saves **38%** of event lookups. Friends: 10%.
- On `rel-avito`, only **0.1%** of ads are shared; categories (88%) and locations (28%) are shared but cost almost nothing to look up. Lists drawn from a single location raise ad sharing only to 3.5%.
- Cold starts (no history at all): 17% on `rel-avito`, 2% on `rel-event`.

So `rel-event` tests the shared-entity coupling, and `rel-avito` tests the ranking-cutoff coupling (§3) with little sharing. That is a built-in contrast for hypothesis H4.

**Natural vs. account-structured call lists.** Random subsets of the test entities give "natural" lists. Lists sampled to share entities (a territory of accounts with several contacts each) give the "account-based selling" condition. That condition is labeled as an intervention, not a population sample.

---

## 5. Method

### M1. Mask-aware relational predictor

This is the RelBench RDL architecture: `HeteroEncoder` (torch_frame per-table encoders), `HeteroTemporalEncoder`, `HeteroGraphSAGE`, and an MLP head. Two changes:

1. **Random revelation during training.** For each mini-batch, draw ρ ~ U(0,1). Reveal each `list` action and each stub with probability ρ, respecting list-before-get. Unrevealed rows become `[MASK]` stubs or are removed.
2. **Per-table `[MASK]` embeddings** replace the encoder output of unfetched rows.

*Ablation:* a mask-naive model (standard full-neighborhood training) evaluated under the same budgets. The expectation, to be measured rather than assumed, is that it degrades and mis-calibrates sharply at low budget.

### M2. Acquisition policies (all share one cache and budget)

| Policy | Idea | Plays the role of |
|---|---|---|
| Random | Uniform over legal actions | Floor |
| Case-local round-robin | Each prospect gets *B/N* lookups: list, then its most-referenced stubs | "Research each prospect individually" (today's default) |
| Uncertainty-first | Spend on the prospect with the highest *p(1−p)* | Classic active feature acquisition |
| Coverage | `get` the stub referenced by the most prospects first | "Enrich the shared account" heuristic; strong and simple |
| Singleton learned scorer (Zhou 2026 style) | GNN scores each action by its own counterfactual gain, independently of other lookups | Closest published method; required baseline |
| Per-target AFA (SEFA/ACO style) | Learned per-prospect acquisition, sharing only the cache | Strongest "research each prospect individually" baseline |
| **Learned batch value (LBV)** | A small GNN over the revealed graph scores each legal action by its predicted drop in *total* batch loss. Trained by regression on oracle gains from training lists. Greedy, recomputed in chunks. | **Proposed method** |
| Oracle-greedy | One-step greedy using the true labels | Diagnostic upper bound; never deployable |
| Full information | Fetch everything | Standard RDL (the RelBench number) |

*Key ablation:* LBV with **batch credit** vs. **per-prospect credit** (count only the prospect that "owns" the action). This tests the claim that joint credit matters when entities are shared.

### M3. Paper-stage extensions (after the course milestone, only if M2 shows headroom)

- **Heterogeneous costs:** `list` is cheap internal CRM data; `get` is costly external enrichment.
- **Nonmyopic value:** two-step lookahead, and a fitted budget-conditioned value baseline (BRiG-style).
- **LLM-agent comparator:** Claude on Amazon Bedrock picks lookups from the schema and the current observations. Does a 1M-parameter GNN planner beat a frontier LLM at deciding what to research?
- **Transfer** across tasks within a database, and across call-list sizes.

---

## 6. Experiments and analyses

| Analysis | What it shows | Rubric line it feeds |
|---|---|---|
| Retrieval-budget frontier: metric vs. lookups per prospect (0, ½, 1, 2, 4, 8, all) for each policy | The headline result | Insights + results; Figures |
| Mask-aware vs. mask-naive predictor across budgets (AUROC and calibration) | Partial-observation training is necessary | Ablation |
| Batch vs. per-prospect credit; natural vs. account-structured lists | When joint planning matters | Ablation; data subsets |
| GraphSAGE vs. GAT vs. LightGBM-on-revealed-features | Is the graph earning its place? | Architecture comparison |
| "What's worth a lookup": which tables and relations each policy fetches, and the value per fetch | Interpretable sales insight ("enrich the account before pulling a third activity page") | Explainability |
| Results by prospect degree, cold-start vs. active, and amount of sharing | Who benefits from research | Performance across subsets |
| Animated rollout of one call list being researched step by step | Intuition | Figures (GIFs score highest) |

Statistics: five seeds for headline numbers, bootstrap confidence intervals over call lists, budgets fixed in advance, and test split used once.

---

## 7. Novelty and the conference path

Full table and wording rules: [RELATED_WORK.md](RELATED_WORK.md) (primary-source check, 2026-10-10).

**Closest work.** *Zhou (2026), "Scout: Budgeted Task-Aware Acquisition of Dynamic Networks"* (arXiv preprint; unrelated to the SCOUT brief agent despite the name) already learns a GNN scorer for which nodes to query under a budget. It scores each query independently, works on homogeneous graphs refreshing stale state, and has no designated target batch. Also close:

- CAMA: cohort-level acquisition, no shared entities.
- KumoRFM-2: random feature and link removal on relational data.
- STEER and LightRDL: compute-motivated context reduction on RelBench.

**Contributions, with honest boundaries.**

- **C1 (protocol + finding):** a retrieval-budgeted evaluation protocol for relational deep learning (chosen, per-row, costed lookups for a call list), and what it reveals about RDL models under partial retrieval. Random-removal robustness and mask training already exist and are cited, not claimed.
- **C2 (method):** **interaction-aware** batch value on heterogeneous relational graphs. The model learns that a `get` is worth more once `list`s reveal how many targets reference the entity, and credits one lookup to every target it informs. It must beat a faithful Zhou-style singleton scorer, a per-target AFA baseline (SEFA/ACO style), and the coverage heuristic, with the same predictor and cache.
- **C3 (extension):** structure-discovery + enrichment actions with heterogeneous costs and nonmyopic value, plus an LLM research-agent comparator. No combination of these for relational prediction was found.

**Deadlines.** Official sites, checked 2026-10-10: LoG 2026 and KDD 2027 cycle 1 have passed; KDD 2027 cycle 2, ICML 2027, and NeurIPS 2027 dates are not yet posted. Re-check before planning a submission.

**Honest venue ladder.**

- The CS224W blog post and Colab are guaranteed outputs.
- A workshop paper (graph learning or agents) is realistic with C1 plus a C2 win on two databases.
- A main-track paper (ICML / KDD / NeurIPS) needs C2 to beat strong acquisition baselines across at least three databases, with a clear mechanism story.

Acceptance is not predictable. Deadlines will be taken from the official sites only.

---

## 8. CS224W deliverables and rubric map

**Project type 1: Real-world application of GNNs** (blog post + Colab). Solo projects are "discouraged; see Ed", so confirm with course staff before the proposal. Deadlines (Fall 2026 handout §4.1, Pacific time): **proposal Tue Oct 20, 11:59pm** (NeurIPS'22 style, ~2 pages excluding references, no abstract); **milestone Tue Nov 10**; **final Thu Dec 10**. The requirement-by-requirement proposal map is in [P8 · Assemble and submit](modules/P8-assemble-and-submit.md#84-requirement-checklist).

| Deliverable | Content |
|---|---|
| **Proposal** (2 pages, NeurIPS style, due Oct 20) | Dataset/task/metric (RelBench, public); why these datasets (headroom + measured sharing); models with a figure and equations (§3, §5); why GNNs fit; preliminary sharing statistics and baseline numbers |
| **Milestone** | Data and graph code, mask-aware predictor, heuristic policies, first LBV results, report draft |
| **Blog post** (50 pts) | Motivation (call list → graph); model explanation tied to the problem; ≥4 analyses from §6; diagrams, frontier plots, rollout GIF; PyG snippets (`HeteroData`, temporal `NeighborLoader`, masking, `HeteroGraphSAGE`, policy scoring) |
| **Colab** (25 pts) | End-to-end on `rel-f1` (fast) and `rel-avito`; every class and function documented |
| **Medium account** (5 pts) | Create early |

---

## 9. AWS setup and budget

- **Region:** `us-west-2`.
- **Compute:** one EC2 GPU instance from the AWS Deep Learning AMI (PyTorch 2.14, Ubuntu 26.04), used for its NVIDIA driver. On top of it, a uv venv pinned to RelBench's tested stack: Python 3.12, torch 2.9.1+cu128, `relbench[example]==3.0.2`, and `pyg-lib` from data.pyg.org. Start at `g6.2xlarge` (1× NVIDIA L4 24 GB, 8 vCPU, 32 GiB RAM; $0.98/hr on-demand). The G-instance vCPU quota defaults to 0, so request 8 first. Resize to a bigger-RAM type only for `rel-amazon` if memory demands it. Access via VS Code Remote-SSH on an Elastic IP.
- **Storage:** one private S3 bucket for checkpoints, results, and figures. RelBench data is re-downloadable and stays on the instance's EBS volume.
- **IAM:** an instance role scoped to that bucket, plus SSM (backup shell access). Bedrock invoke permission is added later for M3.
- **Guardrails:** an AWS Budget with **credits excluded**, so it tracks real consumption (a credits-included budget reads $0 and never alerts). A CloudWatch alarm stops the instance after 60 idle minutes. Stop the instance every time you step away.
- **Spend:** about $1/hr GPU; the debugging dataset also runs on the laptop's CPU environment (P0). The whole course arc fits within roughly $1.5k of the ~$9.8k AWS credits. Azure ($10k) stays in reserve.
- **Bedrock caveat (M3 only):** Claude on Bedrock is billed through AWS Marketplace, and the general promotional-credit terms exclude Marketplace charges unless a credit explicitly authorizes them (AWS Activate credits are a documented exception). Check the credit's "Applicable products" field before running the LLM comparator; Azure credits are the fallback.

---

## 10. Schedule

The step-by-step build lives in [docs/modules](modules/README.md); every step ends with a PASS/STOP check. It runs as two tracks.

**Proposal track** (laptop only; one module per proposal bullet of the handout):

| When | Modules | Output |
|---|---|---|
| Sat Oct 17 | P0–P7 | Environment; dataset cards and metrics; sharing statistics; RDL model with partial observation; Fig. 1; LightGBM and seed-averaged RDL results on `rel-f1`; every proposal section drafted in `proposal/notes.md` |
| Sun Oct 18 – **Tue Oct 20** | P8 | Draft Sunday, polish Monday, submit the 2-page proposal Tuesday |

**Build track** (after the proposal):

| When | Module | Output |
|---|---|---|
| Oct 21 – Oct 25 | B1 Cloud GPU + full-information baselines | AWS guardrails, GPU instance, RDL numbers on `rel-event` and `rel-avito` |
| Oct 26 – Nov 1 | B2 Lookup environment | `list` / `get` actions keyed on `n_id`, shared cache, budgets |
| Nov 2 – Nov 9 | B3 Heuristic policies + frontiers | H1, H3, H4 |
| **Tue Nov 10** | Milestone | Code + report draft |
| Nov 11 – Nov 24 | B4 Learned batch value + Zhou-style baseline | H5 |
| Nov 25 – **Thu Dec 10** | B5 Blog post + Colab | Final submission |

---

## 11. Talking points for the Nooks dinner

- "Building a pre-call brief agent taught me the hard part isn't writing the brief; it's deciding what to look up, for whom, before you know who will pick up."
- "A call list is a graph. Contacts share accounts, so research should be budgeted across the list, not per contact. One account lookup can sharpen five dials."
- "I measure it as a retrieval-budget frontier: accuracy of the call-list ranking per lookup spent. That's the curve a product team can price."
- "The model has to be trained on partial information, because in production it never sees the whole graph."
- "Next: can a small GNN planner beat an LLM agent at deciding what to research? Cheaper, faster, and auditable if it can."

---

## 12. Risks and kill criteria

| Risk | Check | Response |
|---|---|---|
| Little headroom: full information ≈ zero lookups on the chosen task | Tasks were pre-selected by v3-leaderboard GNN–LightGBM gap; re-measure on validation Sunday | Shift weight to `rel-stack/user-engagement` (+28 AUROC points on the leaderboard) |
| Partial-observation training doesn't help (H2) | **Preliminary, 2026-10-17** ([P7](modules/P7-why-appropriate.md)): on `rel-f1` validation, 3 seeds, 20 epochs, rows revealed at random (`drop` mode), the full-trained model degrades gracefully (val AUROC 0.752 ± 0.020 at full information, 0.749 ± 0.010 with 25% of rows, 0.642 ± 0.049 with none) and random-revelation training is no better at any level (0.734 / 0.718 / 0.612). An earlier single-seed comparison that suggested otherwise mixed 10- and 20-epoch runs | Retest with `mask` mode, with policy-chosen lookups (B3), and on `rel-event` / `rel-avito` (B1). If it still doesn't help, report it and use the full-trained model as the predictor; the claim rests on chosen, budgeted retrieval (C1), not on random-removal robustness |
| Coverage heuristic ≈ oracle-greedy | Frontier comparison | The problem is easy there; report it and pivot the paper toward C1 (protocol + finding) |
| LBV doesn't beat the best heuristic | Milestone gate | Report it honestly; the course project is still complete. Try nonmyopic M3 only if oracle headroom exists |
| GPU vCPU quota is 0 on a new account | B1 Level 0 quota check | Request an increase right away; run B1 Levels 2–3 on a CPU instance meanwhile (`rel-f1` is small) |
| Solo-project policy | Ask on Ed | Adjust scope if required |
| Confidential material | SCOUT is private | Never copy SCOUT code, interview notes, or customer data into this public repo |
