"""M2: average train_rdl.py runs over seeds and print a markdown table for the report.

Reads every outputs/<dataset>/<task>/rdl_*_ep*_seed*.json and groups runs by training configuration.
Each cell is the validation AUROC at one revelation level p: mean +/- standard deviation over seeds.
"""
import argparse 
import json
from pathlib import Path 

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")

# or list_auroc, list_ap, list_p@13 (call-list metrics)
parser.add_argument("--metric", default="roc_auc")
args = parser.parse_args()

rows = []
for path in sorted((Path("outputs") / args.dataset / args.task).glob("rdl_*_ep*_seed*.json")):
    run = json.loads(path.read_text())
    a = run["args"]
    config = f"{a['train_reveal']}, {a['hide']}, {a['epochs']} ep"   # one row of the table per configuration
    for p, metrics in run["val"].items():
        rows.append({"config": config, "seed": a["seed"], "p":float(p), "value": metrics[args.metric]})

df = pd.DataFrame(rows)

stats = df.groupby(["config", "p"]).value.agg(["mean", "std"])
seeds = df.groupby("config").seed.unique()
# columns from full information (p = 1.0) down to nothing (p = 0.0)
ps = sorted(df.p.unique(), reverse=True)

print(f"{args.dataset} {args.task}: val {args.metric}, mean +/- sd over seeds\n")
print("| Config | Seeds | " + " | ".join(f"p = {p:g}" for p in ps) + " |")
print("|---|---|" + "---:|" * len(ps))
for config in stats.index.get_level_values("config").unique():
    cells= []

    for p in ps:
        mean, sd = stats.loc[(config, p)]
        cells.append(f"{mean:.3f}" if pd.isna(sd) else f"{mean:.3f} +/- {sd:.3f}")   # sd is NaN for one seed
    print(f"| {config} | {','.join(map(str, sorted(seeds[config])))} | " + " | ".join(cells) + " |")

