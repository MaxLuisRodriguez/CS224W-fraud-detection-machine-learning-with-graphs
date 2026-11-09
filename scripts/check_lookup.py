"""M7: four checks of the lookup environment on one call list, scored by a trained model.

Question it answers: does the lookup environment keep its promises? The environment (src/wtl/lookup.py) is
the game a lookup policy plays. It starts from one call list where only the prospects are visible, and it
offers two kinds of lookup:
  pull(prospect, table): reveal that prospect's own rows in one table (hop 1)
  enrich(table, n_id):   reveal one entity that a pulled row points to (hop 2), in every sampled copy
A lookup costs 1 if it reveals something new and 0 if it was all fetched already (a cache hit).

The four checks:
  (a) zero lookups give the same scores as the model with only the call list revealed (p = 0)
  (b) fetching everything reveals every sampled row and gives the full model's scores
  (c) enriching the same entity twice costs 1, then 0 (the shared cache)
  (d) enriching before any pull is refused: an entity can only be fetched once a pulled row lists it
      ("list before get")

How to run (from the repo root), after scripts/train_rdl.py has saved a model:
    python scripts/check_lookup.py
    python scripts/check_lookup.py --dataset rel-event --task user-ignore

A passing output, in words: (a) and (b) print True, and "every sampled row fetched" is True; (c) prints
"costs 1 then 0"; (d) prints "refused" followed by the list-before-get message. Under (b) it also reports
what full retrieval costs, the paid lookups per table, and the share of listed entities that 2 or more
prospects point to.

Where it fits: milestone module M7 (the lookup environment), after M6 (scripts/check_call_list.py).
tests/test_lookup.py guards the same guarantees as unit tests.
"""
import argparse
from pathlib import Path

import torch

from wtl.data import build_graph, column_stypes, load
from wtl.lookup import CallList, fetch_all, sample_call_list
from wtl.model import RDLModel
from wtl.reveal import random_reveal

# Command-line options: dataset and task, the saved run to load, list size N, and which list to draw.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--run", default="rdl_full_drop_ep20_seed0")
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# Setup: use the GPU if there is one, otherwise the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the data and rebuild the graph exactly as training did.
dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
entity = task.entity_table
# Rebuild the model and load the weights train_rdl.py saved (see check_checkpoint.py).
# .to(device) moves the model to the GPU if there is one; map_location does the same for the loaded weights.
model = RDLModel(data, col_stats).to(device)
path = Path("outputs") / args.dataset / args.task / f"{args.run}.pt"
model.load_state_dict(torch.load(path, map_location=device))
model.eval()
# Sample one call list into one batch (see check_call_list.py) and move it to the model's device.
batch, rows, one_day = sample_call_list(data, task, "val", args.size, args.seed)
batch = batch.to(device)

# (a) Zero lookups: the model sees only the call list itself, the same as random revelation with p = 0.
# CallList(batch, entity) starts a fresh lookup state: only the prospects are in its cache, and cost = 0.
cl = CallList(batch, entity)
with torch.no_grad():
    # The reference: random_reveal with p = 0 hides every row except the targets themselves.
    p0 = torch.sigmoid(model(batch, entity, revealed=random_reveal(batch, entity, 0.0), hide="drop"))

# cl.scores(model): probabilities for the N prospects, seeing only fetched rows (drop mode).
# torch.allclose(..., atol=1e-5): True if the two score vectors match up to tiny rounding differences.
print("(a) zero lookups, same scores as p = 0:", torch.allclose(cl.scores(model), p0, atol = 1e-5))

# (b) Fetch everything: every row revealed, and the scores of the full model.
# fetch_all() makes every pull, then enriches every listed entity, round after round, until nothing new is
# left. It returns the share of listed entities (after all pulls) that 2 or more prospects point to.
full = CallList(batch, entity)
shared = fetch_all(full)
with torch.no_grad():
    # The reference: called without `revealed`, the model sees every sampled row.
    reference = torch.sigmoid(model(batch, entity))
# full.masks(): {table: True/False per sampled row, True = fetched}. m.all() is True only if every row is.
everything = all(m.all().item() for m in full.masks().values())
print("(b) fetch everything, same scores as the full model:", torch.allclose(full.scores(model), reference, atol=1e-5),
      "| every sampled row fetched:", everything)
# full.cost: lookups paid in total; full.n: prospects on the list; full.fetched: {table: lookups paid there}.
print(f"    full retrieval: {full.cost} lookups = {full.cost / full.n:.1f} per prospect; paid lookups by table: {full.fetched}")
print(f"    listed entities that 2+ prospects point to: {shared:.0%}")

# (c) The shared cache: the second enrich of the same entity is free.
cl = CallList(batch, entity)
# legal_pulls(): the (prospect, table) pairs not pulled yet. Take the first one and pull it.
i, table = cl.legal_pulls()[0]
# apply(action, prospect) performs one lookup, charges it to that prospect, and returns its cost (1 or 0).
# A pull action is ("pull", prospect index, table name).
cl.apply(("pull", i, table), i)
# enrich_candidates(): {table: (n_id, prospect, mentions)}, the entities pulled rows point to that are not
# fetched yet. next(iter(d.items())) takes the first (table, candidates) entry of the dict.
ent_table, (n_id, prospect, mentions) = next(iter(cl.enrich_candidates().items()))
# An enrich action is ("enrich", table name, original row id). Do the same one twice: the first must cost 1,
# the second 0 because that entity is already in the cache. int(...) turns a one-number tensor into a Python int.
first = cl.apply(("enrich", ent_table, int(n_id[0])), i)
second = cl.apply(("enrich", ent_table, int(n_id[0])), i)
print(f"(c) pull ({i}, {table}), then enrich {ent_table} {int(n_id[0])} twice: costs {first} then {second}")

# (d) list before get: an entity nobody has pointed to yet can't be enriched.
# A brand-new CallList has pulled nothing, so no row lists that entity yet and apply() should raise ValueError.
# try/except catches the error and prints its message instead of crashing.
fresh = CallList(batch, entity)
try: 
    fresh.apply(("enrich", ent_table, int(n_id[0])))
    print("(d) enrich before any pull: allowed (this is a bug)")
except ValueError as err:
    print("(d) enrich before any pull: refused ->", err)