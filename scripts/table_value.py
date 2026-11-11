"""M9 (E4): what's worth a lookup? Fetch one table of the prospects' own rows at a time and measure the gain.

For each table: pull it for every prospect ("pull only"), then also enrich everything those rows list
("pull + enrich"). Report the AUROC gain over zero lookups, and the gain per 100 lookups spent.

Words used here:
    A prospect's "own rows" in a table are the rows that point straight at that prospect (hop 1), for example
    a user's rows in event_attendees on rel-event. A "pull" fetches all of one prospect's rows in one table,
    for 1 lookup. Those rows mention other entities (for example the events they refer to). An "enrich"
    fetches one such entity, for 1 lookup, and it then becomes visible to every prospect that mentions it.

What this answers:
    Which tables carry the useful signal, and which give the most gain for the lookups they cost? That tells
    a policy what to buy first.

How to run (from the repo root), after train_rdl.py has saved weights:
    python scripts/table_value.py --dataset rel-event --task user-ignore --lists 30

Reads outputs/<dataset>/<task>/<run>.pt (the frozen trained weights).
Writes outputs/<dataset>/<task>/table_value.json (one row per list and table) and prints a markdown table.

How to read the table (one row per table, averaged over call lists):
    "lookups" is the average number of lookups spent; "gain" is the AUROC with those lookups minus the AUROC
    with zero lookups; "per 100 lookups" is 100 * gain / lookups, the value for the price. Lists where AUROC
    is undefined (all prospects have the same label) are left out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wtl.lookup import CallList, sample_call_list
from wtl.runs import load_run

# Command-line options.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
# Which saved training run to load: the name of a .pt file written by train_rdl.py, without ".pt".
parser.add_argument("--run", default="rdl_full_drop_ep20_seed0")
# How many call lists to average over (seeds 0, 1, 2, ...), and how many prospects per list.
parser.add_argument("--lists", type=int, default=10)
parser.add_argument("--size", type=int, default=128)
# Suffix for the output file, so runs on other models don't overwrite table_value.json.
parser.add_argument("--name", default="")
args = parser.parse_args()

# Load the data, build the graph, and load the frozen trained model (the same way as run_frontier.py).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load_run rebuilds the graph and model as the run was trained and switches to evaluation mode.
task, data, model = load_run(args.dataset, args.task, args.run, device)
entity = task.entity_table
out_dir = Path("outputs") / args.dataset / args.task

# One result row per (call list, table).
rows = []
for seed in range(args.lists):
    # Draw one call list: N prospects and their sampled 2-hop neighborhoods, all in one batch.
    batch, _, _ = sample_call_list(data, task, "val", args.size, seed)
    batch = batch.to(device)
    # The baseline: AUROC with zero lookups (only the prospects' own rows are visible).
    zero = CallList(batch, entity).metrics(model)["auroc"]
    # hop1 maps each table of the prospects' own rows to those rows. Try each table on its own.
    for table in sorted(CallList(batch, entity).hop1):
        # A fresh CallList for each table, so lookups from one table never help another.
        cl = CallList(batch, entity)
        # "Pull only": pull this table for every prospect that has rows in it (1 lookup per prospect).
        for i, t in cl.legal_pulls():
            if t == table:
                cl.apply(("pull", i, t), i)
        pull_cost, pull_auroc = cl.cost, cl.metrics(model)["auroc"]
        # "Pull + enrich": now fetch every entity the pulled rows mention. Fetching an entity can make more
        # rows visible that mention further entities, so repeat until nothing new is listed.
        cands = cl.enrich_candidates()
        while cands:                                              # everything this table's rows list
            for ent_table, (n_id, _, _) in cands.items():
                # n_id can repeat (one entry per prospect that mentions the entity), so fetch each entity
                # once. No prospect is named, so the lookup is charged to the list as a whole.
                for nid in n_id.unique().tolist():
                    cl.apply(("enrich", ent_table, nid))
            cands = cl.enrich_candidates()
        rows.append({"seed": seed, "table": table, "zero": zero,
                     "pull_cost": pull_cost, "pull_auroc": pull_auroc,
                     "all_cost": cl.cost, "all_auroc": cl.metrics(model)["auroc"]})
    print(f"list {seed} done", flush=True)

# Summary: average over call lists, one markdown row per table.
print(f"\n{args.dataset}: AUROC gain over zero lookups, mean over {args.lists} lists\n")
print("| Table | Pull only: lookups | gain | per 100 lookups | Pull + enrich: lookups | gain | per 100 lookups |")
print("|---|---:|---:|---:|---:|---:|---:|")
for table in sorted({r["table"] for r in rows}):
    # This table's rows, leaving out lists where AUROC is undefined. Whether AUROC is defined depends only
    # on the list's labels, so checking the zero-lookup value is enough.
    r = [x for x in rows if x["table"] == table and x["zero"] is not None]
    # Average lookups spent: pull only (pc) and pull + enrich (ac).
    pc, ac = np.mean([x["pull_cost"] for x in r]), np.mean([x["all_cost"] for x in r])
    # Average AUROC gain over zero lookups: pull only (pg) and pull + enrich (ag).
    pg = np.mean([x["pull_auroc"] - x["zero"] for x in r])
    ag = np.mean([x["all_auroc"] - x["zero"] for x in r])
    # "per 100 lookups" = 100 * average gain / average lookups.
    print(f"| {table} | {pc:.0f} | {pg:+.3f} | {100 * pg / pc:+.3f} | {ac:.0f} | {ag:+.3f} | {100 * ag / ac:+.3f} |")
# Save the raw per-list rows, so the table can be rebuilt or analyzed further later.
out = out_dir / (f"table_value_{args.name}.json" if args.name else "table_value.json")
out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
print("\nsaved", out)
