"""M6: sample one call list into one subgraph and print its structure: rows, hops, sharing, forward-pass time.

Question it answers: what does one call list look like as a graph, and how much do its prospects share?
A call list is N prospects (default 128) to be ranked on one day. All of their 2-hop neighborhoods are
sampled ONCE into a single batch. That batch is the world the lookup agent explores later (M7).

How to run (from the repo root):
    python scripts/check_call_list.py                                        # rel-f1, for development
    python scripts/check_call_list.py --dataset rel-event --task user-ignore

What it prints, in order:
1. The list: dataset, number of prospects, whether they all share one prediction day, and how many
   have a positive label.
2. Sampled rows per table, as "copies / distinct". Each prospect's neighborhood is sampled separately, so a
   row that several prospects touch is sampled several times. The copies share one original id (n_id).
3. Hop 1: the prospects' own rows (for example a user's event_attendees rows). One "pull" lookup reveals
   one prospect's rows in one table, so the number of possible pulls = (prospect, table) pairs with rows.
4. Hop 2: the entities those rows point to (events, races, friends, ...). These are the candidates for
   "enrich" lookups. For each kind it prints how many are distinct and what share is pointed to by 2 or
   more prospects. A high share means one lookup can help several prospects at once.
5. The time of one forward pass of an (untrained) model over the whole list. This sets how often a lookup
   policy can afford to re-score the list.

A healthy output, in words: distinct <= copies on every line, hop 1 lists the tables whose rows link to
the prospects, and hop 2 lists what those rows point to. On rel-f1 "one prediction day" is False: no
single day has enough drivers, so prospects are drawn from all days and the same driver can appear several
times. That is fine for development only. When the hop-2 table is the prospect table itself, an extra note
gives the share of those entities that are prospects on this list (always visible, never enriched).

Where it fits: milestone module M6 (one call list, one subgraph), before the lookup environment (M7).
"""
import argparse
import time

import torch

from wtl.data import build_graph, column_stypes, load
from wtl.lookup import find_hops, sample_call_list
from wtl.model import RDLModel

# Command-line options. --size is N, the number of prospects; --seed decides which list is drawn.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


# Step 1: load the data and build the whole database as one graph (cached on disk after the first run).
dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))

# The prospect table: drivers on rel-f1, users on rel-event.
entity = task.entity_table
# sample_call_list() draws N prospects scored at one prediction time and samples all their neighborhoods
# into ONE batch (a HeteroData subgraph). The prospects are the first N rows of the entity table in it.
# rows = the task-table rows drawn; one_day = True if all prospects share one prediction time.
batch, rows, one_day = sample_call_list(data, task, "val", args.size, args.seed)

# .y holds the labels the loader attached; the first N entries belong to the prospects.
positives = int(batch[entity].y[: args.size].sum())
print(f"{args.dataset}: {args.size} prospects, one prediction day: {one_day}, positives: {positives}")

# Step 2: rows per table. n_id is each sampled row's ORIGINAL id in the full table, so
# n_id.numel() counts all copies and n_id.unique().numel() counts the distinct rows.
print("\nsampled rows per table (copies / distinct):")
for nt in batch.node_types:
    n_id = batch[nt].n_id
    if n_id.numel() > 0:  # skip tables the list never reached
        print(f" {nt:24s} {n_id.numel():8,d} / {n_id.unique().numel():8,d}")


# Step 3: hop 1. find_hops() reads the list's structure off the sampled edges and returns
#   hop1:  {table: indices of the rows that link directly to a prospect}  (the prospects' own rows)
#   links: [(s, s_rows, t, t_rows)], one entry per edge type: row s_rows[k] of table s links into
#          hop-1 row t_rows[k] of table t. Those s rows are the hop-2 entities.
hop1, links = find_hops(batch, entity, args.size)
print("\nhop 1, the prospects' own rows (one pull = one prospect x one table):")

pulls = 0
for t, r in hop1.items():
    # batch[t].batch: for every sampled row of table t, the index (0 to N-1) of the prospect it was sampled
    # for. Keeping the hop-1 rows r and counting distinct values = how many prospects have rows in table t.
    owners = batch[t].batch[r].unique().numel()
    # Each of those prospects adds one possible pull in this table.
    pulls += owners
    print(f" {t:24s} {r.numel():8,d} rows, {owners:5d} prospects")

print(f"  pull actions: {pulls} ({pulls / args.size:.1f} per prospect)")

# Step 4: hop 2, the entities the hop-1 rows point to, and how many prospects share each one.
print("\nhop 2, entities those rows point to (enrich candidates):")
# The prospects' original ids, used below to spot hop-2 entities that are themselves on the list.
prospect_ids = batch[entity].n_id[: args.size]
for s, s_rows, t, t_rows in links:
    # The hop-2 entities' original ids, one per edge (so the same entity can repeat).
    nid = batch[s].n_id[s_rows]
    # distinct (entity, prospect) pairs
    # torch.stack makes a 2-row tensor: entity ids on top, the prospect each edge belongs to below.
    # .unique(dim=1) removes repeated columns, so each (entity, prospect) pair is kept once.
    pairs = torch.stack([nid, batch[t].batch[t_rows]]).unique(dim=1)

    # unique(return_counts=True) returns (distinct values, how often each occurs); [1] keeps the counts.
    # Applied to the entity row, that is: for each distinct entity, how many prospects point to it.
    reach = pairs[0].unique(return_counts=True)[1]
    note = ""

    # When the hop-2 table is the prospect table (for example a friend who is also a user), report the share
    # of those entities that are on this list. torch.isin(a, b) is True where an element of a is also in b;
    # .float().mean() turns True/False into a share.
    if s == entity:
        note = f", {torch.isin(nid.unique(), prospect_ids).float().mean():.0%} are prospects on the list"

    # (reach >= 2).float().mean(): the share of distinct entities pointed to by 2 or more prospects.
    print(f" {s:14s} via {t:14s} {nid.unique().numel():7,d} distinct, "
        f"{(reach >= 2).float().mean():4.0%} pointed to by 2+ prospects{note}")

# Step 5: time one forward pass over the whole list. The model is untrained; only the time matters here.
model = RDLModel(data, col_stats)
model.eval()

with torch.no_grad():
    start = time.time()  # wall-clock time in seconds
    model(batch, entity)

print(f"\none forward pass over the whole list: {(time.time() - start) * 1000:.0f} ms")