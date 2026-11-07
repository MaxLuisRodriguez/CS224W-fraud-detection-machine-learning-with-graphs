"""M2: reload a saved checkpoint and confirm it scores the validation set like the run that saved it."""
import argparse
from pathlib import Path

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--run", default="rdl_full_drop_ep20_seed0")
args = parser.parse_args()

dataset, task, db = load(args.dataset, args.task)

data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))

val_loader, val_table = make_loader(data, task, "val", batch_size=512)

# same architecture, random weights
model = RDLModel(data, col_stats)
path = Path("outputs") / args.dataset / args.task / f"{args.run}.pt"

model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()

with torch.no_grad():
    scores = torch.cat([torch.sigmoid(model(batch, task.entity_table)) for batch in val_loader]).numpy()

print("loaded", path)
print("val AUROC, full information:", round(task.evaluate(scores, val_table)["roc_auc"], 4))
