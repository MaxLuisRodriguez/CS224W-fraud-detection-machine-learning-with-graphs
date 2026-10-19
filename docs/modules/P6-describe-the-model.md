# P6 · Describe the model with figures and equations (~1.5 h)

> **Handout bullet (§1.3):** "Describe the model(s) (try using figures and equations)."
>
> **Proposal section this feeds:** §3, second part: the partial-observation change, the equations, Fig. 1, and optionally Fig. 2.

P5 built the standard model, which always sees every sampled row. Here you'll let it see only part of the graph (the rows "looked up" so far), write the model as equations that each point to a line of your code, and make the proposal's figures from real data.

## 6.1 What changes: the model sees only what was fetched

Call the set of rows fetched so far the **revealed set** *O*. The call-list targets themselves are always in *O*: you know who is on your list. Everything else in a target's sampled neighborhood is either revealed or not. For now *O* is random: each distinct row is revealed with probability *p*. In the build track (B2) a lookup policy chooses *O* instead.

An unrevealed row can be hidden in two ways:

| Mode | What happens to an unrevealed row | What the model still knows |
|---|---|---|
| `drop` | It's cut out of message passing: every edge touching it is removed | Nothing; the row might as well not exist |
| `mask` | It stays in the graph as a **stub**: its vector is replaced by a learned per-table `[MASK]` vector | That the row exists and which rows reference it, but not its contents or its date |

`mask` is the closer match to the lookup story (P1 §1.3: you know an account exists before you enrich it). `drop` is the simpler baseline. You'll compare both in P7.

## 6.2 `src/wtl/reveal.py`

```python
"""Which rows of a sampled mini-batch the model is allowed to see.

A "revealed" mask holds one boolean per sampled row. Rows are revealed by their original row id (n_id),
so if the same entity was sampled for several targets, all of its copies are revealed together:
fetching a row once makes it visible to every target that touches it.
"""
import torch


def random_reveal(batch, entity_table: str, p: float) -> dict:
    """Reveal each distinct row with probability p; the call-list targets themselves are always revealed."""
    revealed = {}
    for node_type in batch.node_types:
        n_id = batch[node_type].n_id                         # original row id of every sampled row
        distinct = n_id.unique()
        chosen = distinct[torch.rand(distinct.numel(), device=n_id.device) < p]
        revealed[node_type] = torch.isin(n_id, chosen)       # all copies of a chosen row are revealed
    num_targets = batch[entity_table].seed_time.size(0)      # the first rows of the entity table are the targets
    revealed[entity_table][:num_targets] = True
    return revealed
```

This is P5's key point put to work: the coin is flipped once per **distinct** `n_id`, and `torch.isin` copies the answer to every sampled copy of that row.

One simplification to state honestly in the proposal: rows are revealed independently. A race can be revealed while the result row linking it to a driver is hidden, which a real "list, then get" sequence wouldn't allow. In `drop` mode that race is unreachable anyway; in `mask` mode it can pass information through the stub. B2 adds the list-before-get structure.

## 6.3 `src/wtl/model.py` (replaces P5's version)

Two additions: a `[MASK]` vector per table, and `revealed` / `hide` arguments to `forward`. Called without `revealed`, it behaves exactly like P5's model.

```python
"""Relational deep learning model: per-table encoders -> time encoding -> hetero GraphSAGE -> MLP head.

It can also see only part of the graph:
  * hide="drop": unrevealed rows are cut out of message passing (their edges are removed)
  * hide="mask": unrevealed rows stay in the graph as stubs, represented by a learned per-table [MASK] vector
"""
import torch
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.nn import MLP


class RDLModel(torch.nn.Module):
    """Scores target rows from their sampled relational neighborhood (RelBench's baseline design)."""

    def __init__(self, data, col_stats, channels: int = 128, num_layers: int = 2, aggr: str = "sum"):
        super().__init__()
        # One tabular encoder (torch_frame ResNet) per table: a row's columns -> a `channels`-d vector.
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats,
        )
        # Encodes how long before the prediction time each row happened (recency matters).
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels,
        )
        # Message passing over every foreign-key edge type: one SAGEConv per edge type, summed per layer.
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types, edge_types=data.edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        # One learned [MASK] vector per table, for rows that exist but have not been fetched.
        self.mask_emb = torch.nn.ParameterDict(
            {nt: torch.nn.Parameter(torch.zeros(channels)) for nt in data.node_types}
        )
        # Linear readout to one logit per target.
        self.head = MLP(channels, out_channels=1, norm="batch_norm", num_layers=1)

    def forward(self, batch, entity_table: str, revealed: dict = None, hide: str = "drop") -> torch.Tensor:
        seed_time = batch[entity_table].seed_time                 # prediction time of each target
        x_dict = self.encoder(batch.tf_dict)                      # {table: [rows in batch, channels]}
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, t in rel_time.items():
            x_dict[node_type] = x_dict[node_type] + t            # add recency to each row's vector
        edge_index_dict = batch.edge_index_dict

        if revealed is not None and hide == "mask":
            # Stubs: keep the row in the graph, but replace what we would only know by fetching it.
            for node_type, seen in revealed.items():
                stub = self.mask_emb[node_type].expand_as(x_dict[node_type])
                x_dict[node_type] = torch.where(seen.unsqueeze(-1), x_dict[node_type], stub)
        elif revealed is not None and hide == "drop":
            # Unfetched rows send and receive no messages: keep an edge only if both ends are revealed.
            edge_index_dict = {
                (src, rel, dst): ei[:, revealed[src][ei[0]] & revealed[dst][ei[1]]]
                for (src, rel, dst), ei in edge_index_dict.items()
            }

        x_dict = self.gnn(x_dict, edge_index_dict)                # 2 rounds of message passing
        # The first len(seed_time) rows of the entity table are the targets themselves.
        return self.head(x_dict[entity_table][: seed_time.size(0)]).view(-1)
```

Two details worth knowing:

- In `mask` mode the stub replaces the vector *after* the time encoding is added, so a stub loses its date as well as its contents. That matches "known to exist, not yet fetched".
- The `[MASK]` vectors add 9 tables × 128 = 1,152 parameters, so the model now has 5,074,945.

## 6.4 `scripts/check_reveal.py`: check it before training with it

```python
"""P6: check partial observation on one mini-batch before training with it.

Reveals each distinct row with probability p, confirms that all copies of a row (same n_id) got the same answer,
and runs the model once in each hide mode.
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

revealed = random_reveal(batch, task.entity_table, p=args.p)
print(f"reveal p={args.p}:")
for node_type, seen in revealed.items():
    n_id = batch[node_type].n_id
    if n_id.numel() == 0:
        continue
    shown, hidden = n_id[seen].unique(), n_id[~seen].unique()
    split = torch.isin(shown, hidden).sum().item()   # rows revealed in one copy but hidden in another: must be 0
    print(f"  {node_type:14s} {seen.sum().item():4d}/{seen.numel():4d} copies, "
          f"{shown.numel():4d}/{n_id.unique().numel():4d} distinct rows revealed, inconsistent rows: {split}")

model = RDLModel(data, col_stats)
print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")
model.eval()
with torch.no_grad():
    for hide in ("drop", "mask"):
        scores = torch.sigmoid(model(batch, task.entity_table, revealed=revealed, hide=hide))
        print(f"hide={hide}: scores {scores.numpy().round(3)}")
```

```powershell
python scripts\check_reveal.py
```

**PASS** (tested Oct 17; ~20 s):

```text
reveal p=0.5:
  races           305/ 629 copies,   88/ 178 distinct rows revealed, inconsistent rows: 0
  qualifying      171/ 343 copies,  171/ 343 distinct rows revealed, inconsistent rows: 0
  standings       281/ 564 copies,  281/ 564 distinct rows revealed, inconsistent rows: 0
  constructors      5/  19 copies,    3/  10 distinct rows revealed, inconsistent rows: 0
  drivers           8/   8 copies,    8/   8 distinct rows revealed, inconsistent rows: 0
  results         292/ 580 copies,  292/ 580 distinct rows revealed, inconsistent rows: 0

model parameters: 5,074,945
hide=drop: scores [0.517 0.55  0.52  0.536 0.546 0.494 0.542 0.528]
hide=mask: scores [0.517 0.576 0.541 0.526 0.559 0.506 0.538 0.545]
```

What must hold: **inconsistent rows: 0** on every line, all 8 drivers revealed, and roughly half of the distinct rows revealed elsewhere. Exact counts and scores vary between runs: the neighbor sampler doesn't follow `torch.manual_seed` (two runs gave 629 and 632 race rows), and the model is untrained. Try `--p 0`: only the 8 drivers stay revealed.

**STOP:**

- `inconsistent rows` above 0: `random_reveal` is flipping a coin per copy instead of per distinct `n_id`. Compare your `reveal.py` line by line.
- `TypeError: forward() got an unexpected keyword argument 'revealed'`: `model.py` is still P5's version.

## 6.5 The equations, each mapped to its code

Notation: *t* is the target's prediction time, τ(*v*) the timestamp of row *v*, *T*(*v*) its table, *n*(*v*) its original row id, *O* the revealed set of row ids, and *m_v* = 𝟙[*n*(*v*) ∈ *O*]. Bias terms are omitted.

```latex
% (1) Initial embedding of row v: its table's encoder plus a time-since encoding (dated tables only).
h_v^{(0)} = \mathrm{Enc}_{T(v)}(x_v) + \phi_{T(v)}\big(t - \tau(v)\big)

% (2) mask mode: a row not yet fetched enters as its table's learned [MASK] vector.
\tilde h_v^{(0)} = m_v\, h_v^{(0)} + (1 - m_v)\, e^{\mathrm{mask}}_{T(v)}, \qquad m_v = \mathbf{1}[n(v) \in O]

% (3) drop mode: only edges between two revealed rows carry messages.
\mathcal{N}^{O}_r(v) = \{\, u \in \mathcal{N}_r(v) : m_u = m_v = 1 \,\}

% (4) Heterogeneous GraphSAGE layer: one SAGE update per relation r, summed, then LayerNorm + ReLU;
%     only neighbors dated at or before the prediction time are sampled.
h_v^{(\ell+1)} = \mathrm{ReLU}\Big(\mathrm{LN}\Big(\sum_{r \in \mathcal{R}} W_{r,1}^{(\ell)} h_v^{(\ell)}
  + W_{r,2}^{(\ell)} \!\!\sum_{u \in \mathcal{N}_r(v),\, \tau(u) \le t}\!\! h_u^{(\ell)}\Big)\Big)

% (5) Score of call-list target q_i after L = 2 layers.
\hat y_i = \sigma\big(w^\top h_{q_i}^{(L)} + b\big)

% (6) Partial-observation training: a fresh revelation rate rho for every mini-batch.
\min_\theta\; \mathbb{E}_{\rho \sim U(0,1)}\, \mathbb{E}_{O \sim \mathrm{Bern}(\rho)}
  \Big[\, \ell_{\mathrm{BCE}}\big(f_\theta(q_i \mid O),\, y_i\big) \Big]

% (7) The project's objective: choose a lookup policy that ranks the call list best within budget B.
\max_{\pi}\; \mathbb{E}\Big[ M\big( f_\theta(Q \mid O^{\pi}_{B}),\, y \big) \Big]
  \quad \text{s.t.}\quad |O^{\pi}_{B}| - |O_0| \le B

% (8) Batch value of a candidate lookup a: the drop in total loss over the whole list.
V(a \mid O) = \sum_{i \in Q} \Big[ \ell\big(f_\theta(q_i \mid O), y_i\big) - \ell\big(f_\theta(q_i \mid O \cup \{a\}), y_i\big) \Big]
```

| Eq. | Where it lives in your code |
|---|---|
| (1) | `model.py`: `x_dict = self.encoder(batch.tf_dict)`, then `x_dict[node_type] + t` (only tables with a time column get a `t`) |
| (2) | `model.py`: `torch.where(seen.unsqueeze(-1), x_dict[node_type], stub)`; *m_v* keyed by *n*(*v*) is `torch.isin(n_id, chosen)` in `reveal.py` |
| (3) | `model.py`: the `drop` branch, `ei[:, revealed[src][ei[0]] & revealed[dst][ei[1]]]` |
| (4) | `model.py`: `self.gnn(x_dict, edge_index_dict)`, RelBench's `HeteroGraphSAGE` (per-relation `SAGEConv` with `aggr="sum"`, summed across relations, `LayerNorm`, ReLU). The condition τ(*u*) ≤ *t* is enforced by the loader: `time_attr="time"` in `data.py` |
| (5) | `model.py`: `self.head(...)`; σ is `torch.sigmoid` at prediction time (`train_rdl.py`, P7) |
| (6) | `train_rdl.py` (P7): `random_reveal(batch, ..., p=random.random())` and `BCEWithLogitsLoss` |
| (7) | Not built yet: the lookup environment and policies (B2–B3). In P7, *O* is random, not chosen |
| (8) | Not built yet: the learned batch value model (B4) |

In the proposal, present (1)–(6) as what you've built and (7)–(8) as what you'll build. That split is honest and easy to grade.

## 6.6 Fig. 1: one real call list, drawn as a graph

The figure the proposal needs most: what "prospects share entities" looks like on real data. The script draws a random 128-prospect list from the `rel-event` validation day (2012-11-21), keeps the 8 prospects who share the most events with the rest of the list, and draws them with some of their events. It reads the same files `sharing_stats.py` downloaded in P4.

`scripts/fig_call_list.py`:

```python
"""P6: Fig. 1, one real call list from rel-event drawn as a graph.

Draws a random call list of 128 prospects from the user-ignore validation day, keeps the 8 prospects who share
the most events with the rest of the list, and draws them with the events they touched before that day.
Events touched by 2+ of the drawn prospects are highlighted: one lookup about such an event informs all of them.
"""
import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--list_size", type=int, default=128)             # N prospects on the call list
parser.add_argument("--show", type=int, default=8)                    # prospects drawn
parser.add_argument("--max_shared", type=int, default=12)             # shared events drawn, most-shared first
parser.add_argument("--private_per_prospect", type=int, default=2)    # unshared events drawn per prospect
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="proposal/figures/fig1_call_list.pdf")
args = parser.parse_args()

# The same files sharing_stats.py (P4) downloaded, from the same cache, so nothing new is fetched.
val_file = "rel-event/tasks/user-ignore/val.parquet"
root = snapshot_download(
    repo_id="stanford-star/relbench-v1",
    repo_type="dataset",
    allow_patterns=["rel-event/db/event_attendees.parquet", "rel-event/db/event_interest.parquet", val_file],
)
db = f"{root}/rel-event/db".replace("\\", "/")   # DuckDB wants forward slashes, even on Windows
val = pd.read_parquet(f"{root}/{val_file}")
cutoff = val["timestamp"].min().to_pydatetime()    # the validation day: 2012-11-21

# One random call list: N distinct users scored on that day.
rng = np.random.default_rng(args.seed)
call_list = pd.DataFrame({"user": rng.choice(val["user"].unique(), size=args.list_size, replace=False)})

con = duckdb.connect()
con.register("call_list", call_list)
# Every (prospect, event) pair known by the cutoff: attendance records of any status, plus interest clicks.
touches = con.execute(f"""
    SELECT a.user_id AS user, a.event
      FROM '{db}/event_attendees.parquet' a JOIN call_list c ON a.user_id = c.user
     WHERE a.start_time <= $cutoff
    UNION
    SELECT i."user", i.event
      FROM '{db}/event_interest.parquet' i JOIN call_list c ON i."user" = c.user
     WHERE i."timestamp" <= $cutoff
""", {"cutoff": cutoff}).df()

# How many prospects on the whole list touched each event (UNION already removed repeat touches).
touches["n_list"] = touches.groupby("event")["user"].transform("size")
print(f"rel-event, day {cutoff.date()}: call list of {args.list_size}, "
      f"{touches.event.nunique():,} events touched, {(touches.groupby('event').n_list.first() >= 2).mean():.1%} shared")

# Keep the prospects with the most shared events (ties broken by user id, so the draw is reproducible).
shared_counts = touches[touches.n_list >= 2].groupby("user").size().reset_index(name="k")
top = shared_counts.sort_values(["k", "user"], ascending=[False, True]).user.head(args.show).tolist()
name = {u: f"P{i + 1}" for i, u in enumerate(top)}   # anonymous labels: P1 shares the most

sub = touches[touches.user.isin(top)].copy()
sub["n_drawn"] = sub.groupby("event")["user"].transform("size")  # how many of the drawn prospects touched it
candidates = sub[sub.n_drawn >= 2].groupby("event").n_drawn.first().reset_index()
print("events shared by k drawn prospects:", candidates.n_drawn.value_counts().sort_index().to_dict())
# Spread the drawn events across sharing levels (2, 3, ... prospects): one per level, then a second per level, ...
candidates["rank"] = candidates.sort_values("event").groupby("n_drawn").cumcount()
shared = candidates.sort_values(["rank", "n_drawn"]).head(args.max_shared)
# Private events: touched by exactly one prospect on the whole list. A few per prospect, for contrast.
private = sub[sub.n_list == 1].sort_values(["user", "event"]).groupby("user").head(args.private_per_prospect)

# The graph: prospects and events are nodes; "prospect touched event" is an edge.
G = nx.Graph()
G.add_nodes_from(name.values())
for e in shared.event:
    G.add_node(f"e{e}")
for e in private.event:
    G.add_node(f"e{e}")
shown_events = set(shared.event) | set(private.event)
G.add_edges_from((name[u], f"e{e}") for u, e in zip(sub.user, sub.event) if e in shown_events)

# Layout in three rows: private events on top, prospects in the middle, shared events at the bottom.
x_of = {u: x for u, x in zip(top, np.linspace(0, 1, len(top)))}
pos = {name[u]: (x, 0.0) for u, x in x_of.items()}
for u, events in private.groupby("user").event:
    for j, e in enumerate(events):
        pos[f"e{e}"] = (x_of[u] + 0.035 * (j - (len(events) - 1) / 2), 0.45)  # side by side above the prospect
# Order shared events by the average position of the prospects who touched them, so few edges cross.
mean_x = sub[sub.event.isin(shared.event)].assign(x=lambda d: d.user.map(x_of)).groupby("event").x.mean()
for e, x in zip(mean_x.sort_values().index, np.linspace(0, 1, len(mean_x))):
    pos[f"e{e}"] = (x, -1.0)

INK, MUTED, EDGE = "#0b0b0b", "#898781", "#c3c2b7"   # chart ink and hairline grays
PROSPECT, SHARED = "#2a78d6", "#eb6834"              # categorical slots 1 (blue) and 2 (orange)
shared_nodes = [f"e{e}" for e in shared.event]
private_nodes = [f"e{e}" for e in private.event]
shared_edges = [(a, b) for a, b in G.edges if a in shared_nodes or b in shared_nodes]
private_edges = [(a, b) for a, b in G.edges if (a, b) not in shared_edges]

fig, ax = plt.subplots(figsize=(5.5, 3.0))
nx.draw_networkx_edges(G, pos, edgelist=private_edges, edge_color=EDGE, width=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=shared_edges, edge_color=SHARED, width=1.2, alpha=0.6, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=private_nodes, node_color=MUTED, node_size=30, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=shared_nodes, node_color=SHARED, edgecolors="white", linewidths=1.0,
                       node_size=[40 + 30 * n for n in shared.n_drawn], ax=ax)  # bigger = shared by more
nx.draw_networkx_nodes(G, pos, nodelist=list(name.values()), node_color=PROSPECT, edgecolors="white",
                       linewidths=1.5, node_size=420, ax=ax)
nx.draw_networkx_labels(G, pos, labels={n: n for n in name.values()}, font_size=8, font_color="white", ax=ax)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=PROSPECT, markersize=9, label="prospect on the call list"),
    Line2D([], [], marker="o", ls="", color=SHARED, markersize=7, label="event shared by 2+ drawn prospects"),
    Line2D([], [], marker="o", ls="", color=MUTED, markersize=4, label="event no one else on the list touched"),
], loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=7, labelcolor=INK)
ax.axis("off")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"drew {len(top)} prospects, {len(shared_nodes)} of their {len(candidates)} shared events, "
      f"{len(private_nodes)} private events")
print("events per drawn prospect:", {name[u]: int(n) for u, n in sub.groupby("user").size().reindex(top).items()})
print("saved", out)
```

```powershell
python scripts\fig_call_list.py
```

**PASS** (tested Oct 17; ~7 s):

```text
rel-event, day 2012-11-21: call list of 128, 1,078 events touched, 34.3% shared
events shared by k drawn prospects: {2: 136, 3: 44, 4: 45, 5: 21, 6: 10, 7: 1}
drew 8 prospects, 12 of their 257 shared events, 16 private events
events per drawn prospect: {'P1': 230, 'P2': 152, 'P3': 181, 'P4': 100, 'P5': 110, 'P6': 93, 'P7': 75, 'P8': 114}
saved proposal\figures\fig1_call_list.pdf
```

Open the PDF. Three rows: gray private events on top, the eight blue prospects P1–P8 in the middle, and twelve orange shared events at the bottom, each with lines to every drawn prospect who touched it (bigger dot = more prospects). The figure is deterministic: rerunning gives the same picture.

Why the first draft didn't look like this: the 12 *most*-shared events are each touched by 5–7 of the 8 prospects, so they piled up in one blob. The script now takes events from every sharing level (2, 3, … 7 prospects), which shows the range.

**Caption draft.** Adapt it, but keep the counts, because the figure shows a sample:

> *Figure 1: One real call list from `rel-event` (validation day 2012-11-21; 128 random prospects). Shown: the 8 prospects who share the most events with the rest of the list (blue), 12 of the 257 events that at least two of them touched (orange; size = number of these prospects), and 2 events per prospect that no one else on the list touched (gray). One lookup about an orange event informs every prospect linked to it. Across 200 such lists, 34% of touched events are shared (§2).*

**STOP:**

- `duckdb.IOException`: run P4's `sharing_stats.py --dataset rel-event` first; it downloads these files.
- A blank or clipped PDF: paste the printed output and the figure.

## 6.7 Fig. 2 (optional): AUROC vs. share of rows revealed

This figure plots the runs you'll make in P7, so write the script now and **run it at the end of P7**. It averages seeds per training configuration and draws error bars of ±1 standard deviation.

`scripts/fig_reveal_curve.py`:

```python
"""P6 (optional): Fig. 2, validation AUROC vs. the share of rows the model is allowed to see.

Reads the runs train_rdl.py saved in P7 (outputs/<dataset>/<task>/rdl_*_ep*_seed*.json), averages seeds,
and draws one line per training configuration. Error bars are the standard deviation across seeds.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--configs", nargs="*", help="e.g. full_drop_ep10 random_drop_ep20 (default: every run)")
parser.add_argument("--lightgbm", type=float, default=0.661)   # P7's own-row LightGBM val AUROC, for reference
parser.add_argument("--out", default="proposal/figures/fig2_reveal_curve.pdf")
args = parser.parse_args()

rows = []
for path in sorted((Path("outputs") / args.dataset / args.task).glob("rdl_*_ep*_seed*.json")):
    run = json.loads(path.read_text())
    a = run["args"]
    key = f"{a['train_reveal']}_{a['hide']}_ep{a['epochs']}"   # same naming as the output files
    if args.configs and key not in args.configs:
        continue
    label = f"{a['train_reveal']}-trained, {a['hide']}, {a['epochs']} epochs"
    for p, metrics in run["val"].items():
        rows.append({"key": key, "label": label, "p": float(p), "auroc": metrics["roc_auc"]})
df = pd.DataFrame(rows)
stats = df.groupby(["key", "label", "p"]).auroc.agg(["mean", "std", "count"]).reset_index()

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # fixed categorical order, never cycled
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

fig, ax = plt.subplots(figsize=(5.0, 2.8))
groups = list(stats.groupby(["key", "label"]))
for i, (color, ((key, label), g)) in enumerate(zip(COLORS, groups)):
    n = int(g["count"].max())
    dodge = 0.012 * (i - (len(groups) - 1) / 2)   # shift lines apart slightly so error bars don't overlap
    ax.errorbar(g.p + dodge, g["mean"], yerr=g["std"].fillna(0), color=color, lw=2, elinewidth=1, capsize=0,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5,
                label=f"{label} ({n} seed{'s' if n > 1 else ''})")
ax.axhline(args.lightgbm, color=MUTED, lw=1)   # zero-lookup reference: LightGBM on the target's own row
ax.text(1.0, args.lightgbm - 0.004, "LightGBM, own row only", color=MUTED, fontsize=7, ha="right", va="top")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("share of rows revealed to the model (p)", fontsize=8, color=INK)
ax.set_ylabel(f"val AUROC ({args.dataset})", fontsize=8, color=INK)
ax.tick_params(labelsize=7, colors=MUTED)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)                         # gridlines behind the data
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("saved", out)
```

At the end of P7 you'll run it with the configurations that have three seeds, e.g.

```powershell
python scripts\fig_reveal_curve.py --configs full_drop_ep20 random_drop_ep20
```

**PASS:** `saved proposal\figures\fig2_reveal_curve.pdf`; one line per configuration with error bars, a gray LightGBM reference line, and the legend below the axes. The caption must give the dataset (`rel-f1` val, 588 rows), the number of seeds, and say that rows were revealed **at random**, not chosen by a policy.

## 6.8 Optional: a pipeline diagram (don't let it block the proposal)

If you have 30 minutes left after Fig. 1, draw the pipeline in draw.io, PowerPoint, or TikZ and export a PDF to `proposal/figures/fig0_pipeline.pdf`. Recipe, left to right:

1. **Relational database**: four small table icons (`users`, `events`, `event_attendees`, `user_friends`).
2. **Heterogeneous temporal graph**: rows as nodes colored by table, foreign keys as edges, a clock icon for timestamps.
3. **Lookup policy π** (dashed outline, "planned, B2–B4"): picks `list` / `get` actions under budget *B*; output is the revealed set *O*. Draw unfetched rows as hollow nodes (stubs).
4. **Per-table encoders + time encoding**, with "`[MASK]` for stubs" written under it.
5. **Heterogeneous GraphSAGE × 2**.
6. **MLP head → scores** for the call list → **rank → top-*K* calls**.
7. A return arrow from the scores back to the policy, labeled "re-plan".

Solid boxes are built (P5–P7); dashed boxes are planned. Keep it one column wide and grayscale-safe. Fig. 1 is the required figure; this one is a bonus.

## 6.9 Do: write proposal §3, model description

In `proposal/notes.md`:

1. **Partial observation** in 2–3 sentences: the revealed set, `drop` vs. `mask`, and keying by original row id so one fetch serves every target.
2. **Equations (1)–(6)** as built, and **(7)–(8)** as planned. Copy the LaTeX from 6.5.
3. **Fig. 1** with its caption. Refer to it from §1 or §2 too: it's your best motivation figure.
4. The **random-revelation simplification** (6.2), in one sentence.

**Self-check**

1. In `mask` mode, what does the model still know about a stub, and what does it lose?
2. Why are the call-list targets always revealed?
3. Which two equations in your proposal have no code yet, and which build-track modules add them?

**Next: [P7 · Why the model fits the data, and preliminary results](P7-why-appropriate.md)**
