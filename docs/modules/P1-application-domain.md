# P1 · Application domain (~30 min)

> **Handout bullet (§1.3):** "Application domain." Type-1 projects must be "motivated by real-world problems, e.g., an application that a company would actually use."
>
> **Proposal section this feeds:** §1, Application and motivation (~⅓ page).

## 1.1 The problem, in plain words

A sales rep starts the day with a **call list**. On a parallel dialer most dials never connect, but when one does, the rep needs context immediately. A research agent can prepare that context with **lookups**: pulling the prospect's recent activity, enriching their company, checking intent signals. Lookups cost money (enrichment credits), tokens (LLM agents), and latency, so the agent must choose **what to look up, for whom**. This is the problem Nooks' AI research, Signals, and Dialer products sit on.

## 1.2 The graph view

A CRM is a set of linked tables. **Relational Deep Learning** (Fey et al., ICML 2024) turns it into a graph:

- each **row** becomes a **node**, with one node type per table;
- each **foreign key** becomes an **edge** (both directions);
- each row keeps its **timestamp**, and the model only uses rows dated up to the prediction time.

A GNN scores a prospect by passing messages from the rows around it. Two layers reach two hops: the prospect's own activity, and what that activity points to.

## 1.3 Two kinds of lookup

| | **Pull history** `list(prospect, table)` | **Enrich** `get(entity)` |
|---|---|---|
| Returns | The prospect's recent rows in one table | One entity's own row |
| Reveals | New rows, so new connections | Attributes of something already known to exist |
| Sales analogue | "Show me this contact's recent activity" | "Tell me about this company" |
| Benefits | Mostly that prospect | **Every** prospect connected to the entity |

An entity known to exist but not yet enriched is a **stub**.

## 1.4 Why the whole list must be planned at once

1. **Shared entities.** If five prospects touched the same event or company, one enrichment informs all five.
2. **A shared ranking cutoff.** The rep calls the top *K*. A lookup only matters if it can move a prospect across that cutoff, and the cutoff depends on everyone else's scores. This holds **even when prospects share nothing**.

## 1.5 Research question, sub-questions, hypotheses

**RQ.** Given a call list of *N* prospects and a budget of *B* lookups, which lookups should a research agent make so that a graph neural network ranks the list best?

**Sub-questions**

- SQ1 Frontier: how much ranking quality does each lookup buy?
- SQ2 Training: must the GNN be trained on partial neighborhoods to use them well?
- SQ3 Which lookups: which tables are worth fetching first?
- SQ4 Joint planning: when does planning the whole list beat researching prospects one by one?
- SQ5 Learning: can a learned value model beat strong heuristics at choosing lookups?

**Hypotheses.** Predictions, not results; you'll test H1 and H2 in P7.

| # | Hypothesis | Refuted if |
|---|---|---|
| H1 | The frontier is concave: most of the zero-to-full gap is recovered early (≥ 50% of the AUROC gap with ≤ 25% of the rows) | The curve is linear or convex |
| H2 | A GNN trained only on full neighborhoods degrades at low budgets; partial-observation training closes most of that gap | No benefit at low budgets |
| H3 | On `rel-avito`, past search results (`SearchStream`, which holds past clicks) are the most valuable lookup per call | Another table dominates |
| H4 | Batch-aware policies beat case-local ones by more on `rel-event` (dense sharing) than on `rel-avito` (sparse) | Equal advantage, or larger on `rel-avito` |
| H5 | A learned batch-value policy matches or beats the best heuristic and a Zhou (2026)-style singleton scorer | It loses. That is still a publishable frontier finding |

## 1.6 Do: write your proposal §1

Create `proposal/notes.md`, and write 5–7 sentences in your own words covering:

1. the user (an outbound rep or research agent);
2. the cost of lookups;
3. the graph (prospects share entities);
4. the RQ;
5. why GNNs (relational data, message passing over shared entities);
6. what the project produces: a frontier of ranking quality per lookup.

Then practice the 60-second version out loud. It doubles as your Nooks dinner pitch.

**Self-check**

1. Which kind of lookup can benefit several prospects at once, and why?
2. Which of the two couplings exists even with zero shared entities?
3. Why can't a lookup help a prospect with no history?

**Next: [P2 · Which datasets, and are they public?](P2-datasets-and-public.md)**
