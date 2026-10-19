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

