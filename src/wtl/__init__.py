"""Worth the Lookup: budgeted relational retrieval for call-list prioritization.

The question: a sales rep's research agent has a call list of prospects and a limited budget of
database lookups. Which lookups should it spend so that a graph neural network (GNN) ranks the list
best? The modules below build that pipeline from raw tables to scored call lists.

Package map (roughly in pipeline order):
- data.py:     load a RelBench database and turn it into a graph (one node per row, edges from keys)
- text.py:     the GloVe text embedder used while text columns are turned into numbers
- model.py:    the GNN that scores each prospect, and can be limited to the rows fetched so far
- reveal.py:   random "which rows are fetched" masks, used to train the model on partial views
- lookup.py:   the lookup environment (pull / enrich, budget, cache) for one call list
- policies.py: strategies that decide which lookup to make next
- metrics.py:  ranking metrics on call lists (AUROC, AP, Precision@K) and bootstrap intervals
"""