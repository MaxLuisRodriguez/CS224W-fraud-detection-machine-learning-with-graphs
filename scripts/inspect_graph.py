"""P5: build the relational graph for a task, sample one mini-batch, and run the model on it once.

Question it answers: what does the graph look like, and does the whole pipeline run from data to scores?
It is a quick look, not a test: nothing here can fail except by crashing.

How to run (from the repo root):
    python scripts/inspect_graph.py                                         # rel-f1, 8 targets
    python scripts/inspect_graph.py --dataset rel-event --task user-ignore

What it prints, in order:
1. The full graph: every node type (one per table) and every edge type. Each foreign key gives two edge
   types: f2p_<key> from the child row to the parent row it points to, and rev_f2p_<key> going back.
2. One mini-batch: for each table, how many rows were sampled, how many are distinct, and how many are
   duplicates. Duplicates appear because each target's neighborhood is sampled separately, so a row
   shared by several targets is copied once per target.
3. The model's size and its scores for the targets. The model is untrained, so the scores only show that
   the forward pass runs; their values mean nothing yet.

A healthy run prints all three parts without errors. The first run on a dataset also converts every table
into tensors and caches them under ~/.cache/wtl, so it takes longer than later runs.

Where it fits: proposal step P5 (which model), and again in B1 to build rel-event on the GPU instance.
"""
import argparse

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel

# Command-line options. --batch_size is how many targets go into the one mini-batch we look at.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

# =====================================================================================================
# Build the full graph
# =====================================================================================================
# load(): downloads the RelBench data the first time; returns the dataset, the task, and the database.
# column_stypes(): the kind of data in each column (number, category, text, date), which decides its encoding.
# build_graph(): every row becomes a node with a feature vector; every foreign key becomes edges.
# It returns the graph (a PyG HeteroData: a graph with several node and edge types) and per-column statistics.
dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
print(data)  # every node type (table) with its row count, and every edge type (foreign key, both directions)

# =====================================================================================================
# Sample one mini-batch and count duplicates
# =====================================================================================================
# make_loader(): mini-batches of validation targets, each with its 2-hop neighborhood sampled only from rows
# dated on or before that target's prediction time.
loader, _ = make_loader(data, task, "val", batch_size=args.batch_size)
# iter(loader) -> an iterator over mini-batches; next(...) -> just the first batch (a HeteroData subgraph)
batch = next(iter(loader))
# seed_time holds one prediction time per target, so its length is the number of targets in the batch.
num_targets = batch[task.entity_table].seed_time.size(0)       # .size(0) = length of the first dimension
print(f"\nOne mini-batch of {num_targets} targets ({task.entity_table}):")
for node_type in batch.node_types:
    sampled = batch[node_type].num_nodes                         # how many rows of this table were sampled
    if sampled:
        # n_id: each sampled row's ORIGINAL row id in the full table.
        # .unique() -> the distinct ids; .numel() -> how many there are.
        distinct = batch[node_type].n_id.unique().numel()
        print(f"  {node_type:22s} {sampled:6d} sampled rows, {distinct:6d} distinct, "
              f"{sampled - distinct:6d} duplicates (same row sampled for several targets)")

# =====================================================================================================
# Run the (untrained) model once
# =====================================================================================================
# RDLModel: per-table encoders turn rows into vectors, a 2-layer graph network passes messages along the
# edges, and a final layer outputs one score (a logit) per target. Its weights are still random here.
model = RDLModel(data, col_stats)
# model.parameters() yields every weight tensor; p.numel() = how many numbers each holds
print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")
model.eval()                    # evaluation mode: batch norm uses stored statistics, no training behavior
with torch.no_grad():           # don't track gradients: we're only predicting, which saves memory and time
    logits = model(batch, task.entity_table)
# torch.sigmoid: logit -> probability in (0, 1); .numpy() -> NumPy array; .round(3) -> 3 decimals
print("scores (probabilities) for the targets:", torch.sigmoid(logits).numpy().round(3))
