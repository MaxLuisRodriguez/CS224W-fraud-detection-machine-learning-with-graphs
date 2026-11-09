"""The leakage test (tests/test_no_leakage.py) on any dataset: no sampled row may be dated after its target's
prediction time.

Question it answers: can the model see the future? Every target (for example one user on one day) has a
prediction time. The loader must only sample rows dated on or before that time. If a later row slipped in,
the model could read part of the answer from the future and every result would look better than it really
is. That is called leakage.

The unit test runs on the small rel-f1 graph; this script runs the same check on any dataset (rel-event in
particular), over several training and validation mini-batches, and reports each table separately. The
targets' own rows are counted apart from the neighbors the sampler added, because the sampler always
includes a target's own row, whatever its date.

How to run (from the repo root):
    python scripts/check_leakage.py                                         # rel-f1, 5 batches per split
    python scripts/check_leakage.py --dataset rel-event --task user-ignore

A passing output, in words: for each split, one line per (table, role) with the rows checked and how many
are dated after the target's prediction time. Every "sampled neighbor" line must show 0. A "target's own
row" line is the target itself, which the loader includes on purpose, so a count there describes the data
rather than a sampler bug. Tables without a time column are not listed: they cannot leak by date.
"""
import argparse

from wtl.data import build_graph, column_stypes, load, make_loader

# Command-line options. --batches keeps the check quick by looking at only the first few mini-batches.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--batches", type=int, default=5)   # mini-batches checked per split
args = parser.parse_args()

# Load the data and build the graph with the same code training uses. Column statistics are not needed here.
dataset, task, db = load(args.dataset, args.task)
data, _ = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
# The table whose rows are the prediction targets (drivers on rel-f1, users on rel-event).
entity = task.entity_table

# Check both splits: training (shuffled, as during training) and validation.
for split in ["train", "val"]:
    loader, _ = make_loader(data, task, split, shuffle=(split == "train"))
    counts = {}                                                  # (table, role) -> [rows, rows dated after t]
    for i, batch in enumerate(loader):
        if i == args.batches:
            break
        seed_time = batch[entity].seed_time                      # each target's prediction time
        n_targets = seed_time.size(0)                            # targets are the first rows of the entity table
        for node_type in batch.node_types:
            store = batch[node_type]                             # the sampled rows of one table, with their attributes
            if "time" not in store or store.num_nodes == 0:
                continue                                         # tables without dates can't leak the future
            # For every sampled row: is it dated after the prediction time of the target it was sampled for?
            # seed_time[store.batch] looks up that target's time once per row, so the comparison is row by row.
            late = store.time > seed_time[store.batch]           # store.batch: which target each row was sampled for
            # In the target table, the first n_targets rows are the targets themselves and the rest are rows of
            # the same table reached as neighbors; count the two roles apart. slice(a, b) is the object behind
            # the [a:b] syntax, and slice(n, None) means [n:].
            if node_type == entity:
                parts = {"target's own row": slice(0, n_targets), "sampled neighbor": slice(n_targets, None)}
            else:
                parts = {"sampled neighbor": slice(0, None)}
            for role, sl in parts.items():
                # setdefault returns the running [rows, late rows] pair for this key, creating [0, 0] the first time.
                c = counts.setdefault((node_type, role), [0, 0])
                c[0] += int(late[sl].numel())                    # rows checked
                c[1] += int(late[sl].sum())                      # rows dated after the prediction time (True = 1)
    # One line per (table, role), sorted by table name.
    print(f"{split}, {args.batches} batches:")
    for (node_type, role), (rows, future) in sorted(counts.items()):
        print(f"  {node_type:16s} {role:17s} {rows:8,d} rows, {future:5d} dated after the target's prediction time")
