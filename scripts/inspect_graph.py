"""P5: build the relational graph for a task, sample one mini-batch, and run the model on it once."""
import argparse

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

# =====================================================================================================
# Build the full graph
# =====================================================================================================
dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
print(data)  # every node type (table) with its row count, and every edge type (foreign key, both directions)

# =====================================================================================================
# Sample one mini-batch and count duplicates
# =====================================================================================================
loader, _ = make_loader(data, task, "val", batch_size=args.batch_size)
# iter(loader) -> an iterator over mini-batches; next(...) -> just the first batch (a HeteroData subgraph)
batch = next(iter(loader))
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
model = RDLModel(data, col_stats)
# model.parameters() yields every weight tensor; p.numel() = how many numbers each holds
print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")
model.eval()                    # evaluation mode: batch norm uses stored statistics, no training behavior
with torch.no_grad():           # don't track gradients: we're only predicting, which saves memory and time
    logits = model(batch, task.entity_table)
# torch.sigmoid: logit -> probability in (0, 1); .numpy() -> NumPy array; .round(3) -> 3 decimals
print("scores (probabilities) for the targets:", torch.sigmoid(logits).numpy().round(3))
