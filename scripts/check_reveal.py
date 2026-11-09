"""M1: check partial observation on one mini-batch before training with it.

Question it answers: when we hide part of the graph from the model, does the hiding behave correctly?
Hiding rows is how the project simulates a research agent that has not looked them up yet.

Background in two sentences. The loader samples each target's neighborhood separately, so a row that
several targets share shows up as several copies, all with the same original row id (n_id). Reveals are
decided per n_id, so all copies of one row must be revealed or hidden together.

What it does:
1. Reveal each distinct row with probability p; every copy of a row (same n_id) must get the same answer.
2. With every row revealed, both hide modes must reproduce the full model exactly.
   hide="drop" cuts hidden rows out of the graph; hide="mask" keeps them but swaps their vector for a
   learned per-table [MASK] vector.
3. Score the batch in each hide mode.

How to run (from the repo root):
    python scripts/check_reveal.py              # rel-f1, 8 targets, p = 0.5
    python scripts/check_reveal.py --p 0        # nothing revealed except the targets themselves

A passing output, in words:
- Every table line ends with "inconsistent rows: 0".
- Every target is revealed (the targets are the call list itself, which is always known).
- Roughly a share p of the distinct rows of every other table is revealed.
- Both "same as full model" lines print True.
- The final scores come from an untrained model, so their values mean nothing. They only show that the
  forward pass runs in both modes.
One case to know about: random_reveal always reveals the targets, even when their own coin flip said
"hidden". If a target's row is also sampled as another target's neighbor (possible when, say, users are
friends of other users on the list), that other copy can stay hidden, and the target table would then
show a nonzero count. Every other table must show 0.

Where it fits: milestone module M1 (the model that can hide rows), right after src/wtl/model.py and
src/wtl/reveal.py are written.
"""
import argparse

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel
from wtl.reveal import random_reveal

# Command-line options. The defaults use rel-f1, the small development dataset, so this runs on a laptop.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--batch_size", type=int, default=8)  # how many targets in the one mini-batch
parser.add_argument("--p", type=float, default=0.5)  # chance that each distinct row is revealed
args = parser.parse_args()

# Fix the random seed so the coin flips and the untrained model's random weights are the same on every run.
torch.manual_seed(0)

# Step 0: load the data, build the graph, and take one mini-batch of validation targets.
# load(): downloads the RelBench data the first time; returns the dataset, the task, and the database.
# column_stypes(): the kind of data in each column (number, category, text, date), which decides its encoding.
# build_graph(): every table row becomes a node, every foreign key a pair of edge types (one per direction).
dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
# make_loader(): mini-batches of targets, each with its 2-hop neighborhood sampled from rows dated on or
# before that target's prediction time. next(iter(loader)) takes just the first mini-batch.
loader, _ = make_loader(data, task, "val", batch_size=args.batch_size)
batch = next(iter(loader))
# The table whose rows are scored (drivers on rel-f1, users on rel-event).
entity = task.entity_table

# 1. Every copy of a row must get the same answer.
# random_reveal() returns {table name: one True/False per sampled row, True = revealed}.
# It flips one coin per distinct n_id, then always reveals the targets themselves.
revealed = random_reveal(batch, entity, p=args.p)
print(f"reveal p={args.p}:")
for node_type, seen in revealed.items():
    # n_id: each sampled row's ORIGINAL row id in the full table. Copies of one row share the same n_id.
    n_id = batch[node_type].n_id
    if n_id.numel() == 0:
        # this table wasn't reached by the batch
        continue

    # n_id[seen] keeps the ids of revealed copies; n_id[~seen] the ids of hidden copies (~ flips True/False).
    # .unique() collapses repeated ids, so shown and hidden are sets of distinct rows.
    shown, hidden = n_id[seen].unique(), n_id[~seen].unique()
    # rows revealed in one copy, hidden in another
    # torch.isin(a, b): True for each element of a that also appears in b. A row in both sets is
    # inconsistent. .sum() counts the True values; .item() turns the one-number tensor into a Python int.
    split = torch.isin(shown, hidden).sum().item()

    # Revealed copies / all copies, then revealed distinct rows / all distinct rows, then the inconsistent count.
    print(f" {node_type:14s} {seen.sum().item():4d}/{seen.numel():4d} copies, "
          f"{shown.numel():4d}/{n_id.unique().numel():4d} distinct rows revealed, inconsistent rows: {split}")


# Build the model. Its weights are random (untrained): this script checks the plumbing, not accuracy.
model = RDLModel(data, col_stats)
# eval(): evaluation mode, so batch norm uses its stored statistics instead of the current batch's.
model.eval()
# model.parameters() yields every weight tensor; p.numel() counts the numbers in each; the sum is the model size.
print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")

# no_grad(): we only predict here, so skip the gradient bookkeeping (saves memory and time).
with torch.no_grad():
    # 2 Hiding nothing must change nothing
    # everything: True for every sampled row of every table. batch[nt].num_nodes = rows sampled from table nt.
    everything = {nt: torch.ones(batch[nt].num_nodes, dtype=torch.bool) for nt in batch.node_types}
    # The reference: called without `revealed`, the model sees every row. It returns one logit per target.
    full = model(batch, entity)

    for hide in ("drop", "mask"):
        # With every row revealed, "drop" removes no edges and "mask" replaces no vectors, so both must
        # match the reference. torch.allclose(a, b, atol=1e-5): True if every pair differs by at most ~1e-5.
        same = torch.allclose(full, model(batch, entity, revealed=everything, hide=hide), atol=1e-5)

        print(f"hide={hide}, every row revealed, same as full model: {same}")

    # 3. scores seeing only the revealed share (untrained model, so the values mean nothing yet).
    # torch.sigmoid turns each logit into a probability between 0 and 1; .numpy().round(3) prints 3 decimals.
    # Both modes see the same revealed rows but treat the hidden ones differently, so their scores differ.
    for hide in ("drop", "mask"):
        scores = torch.sigmoid(model(batch, entity, revealed=revealed, hide=hide))
        print(f"hide={hide}: scores {scores.numpy().round(3)}")

