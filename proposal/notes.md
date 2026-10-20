# Proposal notes

## §1 Application domain

**The user and the pain** (step 1): Who uses this, and what are they trying to do before a call?
- an outbound sales rep, or the AI research agent working for them, has a call list of ~100 + prospects and minutes to prepare. On a parallel dialer most calls never connect; when one does, the rep needs relevant context on screen immediately. The pain isn't writing the brief; it's deciding what to research, for whom, before knowing who'll pick up.

**What a lookup costs** (step 1): Name three costs, with a sales example of each.
Money: enrichment providers charge per record. Tokens and compute: an LLM agent pays to read and summarize each source
(3) latency: research that isn't finished when the dialer connects is worthless. Also, a brief with 20 facts is worse than one with the 3 that matter, so more lookups isn't automatically better.

**Why it's a graph problem** (step 1): What are the nodes, the edges, and the time rule, in one sentence each?
-Nodes: ever row of every table (prospects, events, accounts, and the activity rows linking them).
- Edges: every foreign key, meaning one row pointing to another , stored in both directions.
-Time rule: a prediction made at time t may only use rows dataed up to t. Anything later is the future, and using it is label leakage.
Deeper: the graph isn't hand -designed. It falls out of the database schema, which is why the approach generalizes to any CRM.

**Two kinds of lookup** (step 2): Pull history vs. enrich: what does each reveal, and who benefits?
- Pull history (list) fetches a prospect's recent rows in one table. It reveals what they did and which entities they touched, which become stubs. It mainly benefits that prospect.
- Enrich (get) fetches a known entity's own row. It reveals what that entity is, and benefits every prospect connected to it.

Deeper: enrich depends on pull (so you can't enrich a stub you haven't disocvered), so lookups have order dependencies.

**Reason 1, shared entities** (step 3): Explain with the event-812 example.
Ana and Ben both responded to event 812. after pulling both histories, 812 is a stub on both sides, and one enrichment updates both predictions. A case-local planner credits that lookup to one prospect, undervalues it, and might fetch it twice.

Deeper: how much this matters is an empirical question. 34% of events are shared on rel-event, but only 0.1% of ads on rel-avito.

**Reason 2, the ranking cutoff** (step 3): Explain with the K = 2 example. Why does it hold even with no sharing?
with K = 2 slots and socres Ana 0.80, Ben 0.55, Cara 0.53, Dev 0.20, the cutoff sits between Ben and Cara. Researching them could swap who gets called; researching Ana or Dev almost certainly won't. If one lookup drops Ben to 0.40, the cutoff moves and the best next lookup changes. The value of researching one prospect depends on everyone's scores, so planning must be joint even when nobody shares anything.
Deeper: this why Preicision@K is a natural metric here. It only cares about getting the top K right.

**Research question** (step 4): Write it from memory.
Given a call list of N prospects and a budget of B lookups, which lookups should a research agent make so that a graph neural network ranks the list best?

**What we'll produce** (step 4): The frontier: what it is, and why a product team can price it.
- H1: the frontier is concave, so early lookups buy most of the accuracy.

**Hypotheses, one line each, with today's status** (step 4):
- H1: The frontier is concave, so early lookups buy most of the accuracy.
- H2: training on partial neighborhoods helps at low budgets.
- H3: On rel-avito, past search results (past clicks) are the most valuable lookup.
- H4: batch-aware planning beats case-local by more on rel-event (dense sharing) than rel-avito.
- H5: the learned batch value beats the best heuristic and the Zhou-style singleton scorer.


## §2 Datasets (opening)

**Which datasets:** (name the 3 dataset/task pairs and the role of each: dense sharing, large headroom, development)

**Public availability:** (where they're hosted, that no login is needed, the original sources, and that you don't re-host)

## §2.1 Dataset, task, and metric

### Tasks at a glance
rel-event user-ignore : val positive rate 11.1%
rel-avito user-clicks : val positive rate 3.5%
rel-f1 driver-top3  : val positive rate 20.2%

### Schemas
- rel-event: prospects are users. Attendance records (11.2 million (M)), friendships (30.4M), interest clicks (15k), and created events point at them. Two hops away sit the events they responded to and their friends. Friendships have no timestamp, so they're always visible. 

-rel-avito: prospects are users. Searches (2.6M), ad visits (6.5M) and phone-number requests (303k) point at them. Two hops away sit the ads they visited or requested, the cateogires and locations of their searches, and the ads shown in their searches.

- rel-f1: prospects are drivers. Results, standings and qualifiying rows point at them. Two hops away sit races and constructors; circuits are three hops. out of a 2-layer GNN's reach.

### Splits
- Splits are by time. Train prediction dates are spaced one horizon apart up to the validation cutoff: 22 dates on rel-event, 3 on rel0avit, 59 on rel-f1.
- Validation is a single day on both primary dataset: 2012-11-21 with 2.013 users. and 2015-05-08 with 21,183 users. That's one day's call list, and lists of 128 are sampled from it.
- Test is evaluated once, at the end.

### Metrics
- AUROC: RelBench's official metric, comparable to the leaderboard; 0.5 is random.
- Average precision: how concentrated the positives are at the top; random equals the positive rate.
- Precision@K over 200 random lists of N = 128, with K = ceiling 0.1*N = 13 here, the calls a rep makes first. ALways reported next to the base rate.
- Headline: the budget frontier, each metric against lookups per prospect, one curve per policy, plus the area under it.

### Why this model fits
relational data is used directly through message passing over foreign keys, and the GNN beats a target-row-only model. Temporal sampling prevents label leakage, verified by a unit test. Shared entities carry the same row id across targets, so one fetch reveals them to all. The same trained model runs at any level of retrieval. Limit: so far rows are hidden at random, on one small datatset.

## §4 Evaluation plan and preliminary results

### Evaluation plan
Budget frontiers for random, case-local round-robin, uncertainty-first, coverage, a Zhou-style singleton scorer, the learned batch value, oracle-greedy, and full information. Ablations: training regime, stubs vs. removal, batch vs. per-prospect credit. Architectures: GraphSAGE, GAT, LightGBM. Subsets: cold starts, sharing level, list size. H1--H5 one line each. Final results with five seeds.

### Preliminary results
Table 2's three columns, labeled "rel-f1 validation, 3 seeds, 20 epochs, rows revealed at random". H1 consistent (25% keeps ~97% of the gap); H2 not supported here; LightGBM wins at zero lookups through identity features; runs vary by 0.021--0.05, hence the seeds.