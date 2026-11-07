"""M1: check partial observation on one mini-batch before training with it.

1. Reveal each distinct row with probability p; every copy of a row (same n_id) must get the same answer.
2. With every row revealed, both hide modes must reproduce the full model exactly.
3. Score the batch in each hide mode.
"""
import argparse

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel
from wtl.reveal import random_reveal

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--p", type=float, default=0.5)
args = parser.parse_args()

torch.manual_seed(0)

dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
loader, _ = make_loader(data, task, "val", batch_size=args.batch_size)
batch = next(iter(loader))
entity = task.entity_table

# 1. Every copy of a row must get the same answer.
revealed = random_reveal(batch, entity, p=args.p)
print(f"reveal p={args.p}:")
for node_type, seen in revealed.items():
    n_id = batch[node_type].n_id
    if n_id.numel() == 0:
        # this table wasn't reached by the batch
        continue

    shown, hidden = n_id[seen].unique(), n_id[~seen].unique()
    # rows revealed in one copy, hidden in another
    split = torch.isin(shown, hidden).sum().item()

    print(f" {node_type:14s} {seen.sum().item():4d}/{seen.numel():4d} copies, "
          f"{shown.numel():4d}/{n_id.unique().numel():4d} distinct rows revealed, inconsistent rows: {split}")


model = RDLModel(data, col_stats)
model.eval()
print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    # 2 Hiding nothing must change nothing
    everything = {nt: torch.ones(batch[nt].num_nodes, dtype=torch.bool) for nt in batch.node_types}
    full = model(batch, entity)

    for hide in ("drop", "mask"):
        same = torch.allclose(full, model(batch, entity, revealed=everything, hide=hide), atol=1e-5)

        print(f"hide={hide}, every row revealed, same as full model: {same}")

    # 3. scores seeing only the revealed share (untrained model, so the values mean nothing yet).
    for hide in ("drop", "mask"):
        scores = torch.sigmoid(model(batch, entity, revealed=revealed, hide=hide))
        print(f"hide={hide}: scores {scores.numpy().round(3)}")

