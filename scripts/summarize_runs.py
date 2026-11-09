"""M2: average train_rdl.py runs over seeds and print a markdown table for the report.

What this answers:
    Training has randomness (weight initialization, batch order, sampled neighborhoods), so one run can be
    lucky or unlucky. This script groups runs that share a training setup and reports the mean and the
    spread over seeds, so a difference between setups can be judged against run-to-run noise.

How to run (from the repo root), after train_rdl.py has been run for a few seeds:
    python scripts/summarize_runs.py --dataset rel-event --task user-ignore
    python scripts/summarize_runs.py --dataset rel-event --task user-ignore --metric list_auroc

Reads every outputs/<dataset>/<task>/rdl_*_ep*_seed*.json and groups runs by training configuration.
Writes nothing; it prints the table.

How to read the table:
    One row per configuration (train_reveal, hide, epochs), with the seeds that were averaged.
    Each cell is the validation metric (AUROC by default) at one revelation level p: mean +/- standard
    deviation over seeds. Columns run from p = 1 (every row visible) down to p = 0 (only the targets' own
    rows). A cell without "+/-" came from a single seed.
"""
import argparse 
import json
from pathlib import Path 

import pandas as pd

# Command-line options: which dataset and task folder to read.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")

# Which metric fills the cells. The default "roc_auc" is RelBench's AUROC over the whole validation table;
# or list_auroc, list_ap, list_p@13 (call-list metrics)
parser.add_argument("--metric", default="roc_auc")
args = parser.parse_args()

# Step 1: read every saved run into one long table with one row per (run, reveal level).
rows = []
# glob finds the files that match train_rdl.py's naming pattern; sorted() keeps the order stable.
for path in sorted((Path("outputs") / args.dataset / args.task).glob("rdl_*_ep*_seed*.json")):
    run = json.loads(path.read_text())
    # "args" holds the command-line settings that produced this run.
    a = run["args"]
    config = f"{a['train_reveal']}, {a['hide']}, {a['epochs']} ep"   # one row of the table per configuration
    # "val" maps each reveal level p (saved as a string key) to that level's metrics.
    for p, metrics in run["val"].items():
        rows.append({"config": config, "seed": a["seed"], "p":float(p), "value": metrics[args.metric]})

df = pd.DataFrame(rows)

# Step 2: mean and standard deviation over seeds for each (configuration, p) pair.
stats = df.groupby(["config", "p"]).value.agg(["mean", "std"])
# Which seeds each configuration has, shown in the table so you can see what was averaged.
seeds = df.groupby("config").seed.unique()
# columns from full information (p = 1.0) down to no lookups (p = 0.0: only the targets' own rows)
ps = sorted(df.p.unique(), reverse=True)

# Step 3: print the markdown table: a title, the header, an alignment row ("---:" right-aligns numbers),
# then one row per configuration.
print(f"{args.dataset} {args.task}: val {args.metric}, mean +/- sd over seeds\n")
print("| Config | Seeds | " + " | ".join(f"p = {p:g}" for p in ps) + " |")
print("|---|---|" + "---:|" * len(ps))
for config in stats.index.get_level_values("config").unique():
    cells= []

    for p in ps:
        # .loc[(config, p)] picks one row of the stats table: its mean and standard deviation.
        mean, sd = stats.loc[(config, p)]
        cells.append(f"{mean:.3f}" if pd.isna(sd) else f"{mean:.3f} +/- {sd:.3f}")   # sd is NaN for one seed
    print(f"| {config} | {','.join(map(str, sorted(seeds[config])))} | " + " | ".join(cells) + " |")

