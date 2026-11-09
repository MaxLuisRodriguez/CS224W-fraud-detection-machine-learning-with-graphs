"""M2: reload a saved checkpoint and confirm it scores the validation set like the run that saved it.

Question it answers: can the trained weights come back from disk? Later modules (the lookup environment
and the lookup policies) rebuild the model and load these weights instead of retraining, so this must work.

How to run (from the repo root), after scripts/train_rdl.py has saved a run:
    python scripts/check_checkpoint.py
    python scripts/check_checkpoint.py --dataset rel-event --task user-ignore --run rdl_full_drop_ep20_seed0

--run is the saved file's name without ".pt". train_rdl.py writes outputs/<dataset>/<task>/<run>.pt
(the weights) next to <run>.json (the metrics it measured).

A passing output, in words: it prints the path it loaded and one validation AUROC with full information.
That number should be close to, but not always exactly equal to, the "reveal p=1.00" line train_rdl.py
printed for the same run: the loader samples fresh neighborhoods on every pass, so scores move a little.
An error while loading means the model's architecture changed between saving and loading (for example,
src/wtl/model.py was edited in between).

Where it fits: milestone module M2 (train once, save the weights).
"""
import argparse
from pathlib import Path

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel

# Command-line options: which dataset and task, and which saved run to load.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--run", default="rdl_full_drop_ep20_seed0")
args = parser.parse_args()

# Step 1: rebuild what training used: the data, the graph, and the validation loader.
# The model's shape depends on the graph (its tables and columns), so the graph comes first.
dataset, task, db = load(args.dataset, args.task)

data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))

# 512 targets per mini-batch. val_table is the validation task table in loader order (entity ids,
# prediction times, labels); task.evaluate needs it at the end to line scores up with labels.
val_loader, val_table = make_loader(data, task, "val", batch_size=512)

# Step 2: build the model and load the saved weights into it.
# same architecture, random weights
model = RDLModel(data, col_stats)
# The "/" operator on Path objects joins folder names, e.g. outputs/rel-f1/driver-top3/<run>.pt
path = Path("outputs") / args.dataset / args.task / f"{args.run}.pt"

# torch.load(path, map_location="cpu") reads the saved {parameter name: tensor} dict; map_location puts
# tensors saved on a GPU onto the CPU. load_state_dict() then overwrites the random weights with the saved
# ones, matching them by name, and raises an error if any name or shape does not match.
model.load_state_dict(torch.load(path, map_location="cpu"))
# eval(): evaluation mode, so batch norm uses its stored statistics instead of the current batch's.
model.eval()

# Step 3: score every validation target with full information (nothing hidden).
# For each mini-batch: model -> one logit per target; sigmoid -> a probability between 0 and 1.
# torch.cat joins the per-batch results into one tensor; .numpy() converts it for RelBench.
# no_grad(): we only predict, so skip the gradient bookkeeping.
with torch.no_grad():
    scores = torch.cat([torch.sigmoid(model(batch, task.entity_table)) for batch in val_loader]).numpy()

# task.evaluate(scores, table) returns a dict of RelBench's official metrics. AUROC is the chance that a
# random positive is scored above a random negative (0.5 = guessing, 1.0 = perfect ranking).
print("loaded", path)
print("val AUROC, full information:", round(task.evaluate(scores, val_table)["roc_auc"], 4))
