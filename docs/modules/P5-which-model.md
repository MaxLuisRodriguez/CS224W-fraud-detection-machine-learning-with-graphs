# P5 · Which graph ML model? (~1 h)

> **Handout bullet (§1.3):** "Which graph ML model are you planning to use? (Be sure to cite!)"
>
> **Proposal section this feeds:** §3, first paragraph.

You'll build the standard Relational Deep Learning model, run it once on a real mini-batch, and see the one property of its sampler that shapes everything later: each target gets its own copy of the rows it shares with other targets.

## 5.1 The model, in one paragraph

**Relational Deep Learning** (RDL; Fey et al., ICML 2024) turns a database into a heterogeneous temporal graph: one node per row, one edge type per foreign key (both directions), a timestamp per row. RelBench (Robinson et al., NeurIPS 2024) ships the reference model, which you'll reproduce:

1. **Per-table encoders.** A `torch_frame` ResNet per table turns a row's columns (numbers, categories, text embedded with GloVe) into a 128-d vector.
2. **Temporal encoder.** Adds an encoding of how many days before the prediction time each row happened.
3. **Heterogeneous GraphSAGE** (Hamilton et al., NeurIPS 2017). Two layers; each layer runs one SAGE convolution per edge type, sums the results across edge types, then applies LayerNorm and ReLU.
4. **MLP head.** One logit per target, so one probability per prospect.

Mini-batches come from PyG's temporal `NeighborLoader` (Fey & Lenssen, 2019): for each target, up to 128 neighbors per edge type at hop 1 and 64 at hop 2, and only rows dated at or before that target's prediction time.

Citations, with links:

- Fey et al. 2024, *Position: Relational Deep Learning – Graph Representation Learning on Relational Databases*, ICML 2024: [proceedings.mlr.press/v235/fey24a.html](https://proceedings.mlr.press/v235/fey24a.html)
- Robinson et al. 2024, *RelBench*: already in `refs.bib` (P2)
- Hamilton, Ying & Leskovec 2017, *Inductive Representation Learning on Large Graphs*: [arXiv 1706.02216](https://arxiv.org/abs/1706.02216)
- Fey & Lenssen 2019, *Fast Graph Representation Learning with PyTorch Geometric*: [arXiv 1903.02428](https://arxiv.org/abs/1903.02428)

## 5.2 `src/wtl/text.py`

RelBench tables hold text (driver names, ad titles, event descriptions). `torch_frame` needs a function that turns a list of strings into a tensor; RelBench's examples use averaged 300-d GloVe word vectors, which are fast and run on the CPU.

```python
"""Text embedder used when RelBench text columns are turned into tensors."""
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    """Callable that maps a batch of strings to 300-d averaged GloVe word vectors (fast, CPU-friendly)."""

    def __init__(self, device: Optional[str] = None):
        # Downloads the model (~480 MB) from Hugging Face on first use, then reuses the local cache.
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", device=device
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        # Text columns can hold None/NaN; the tokenizer needs real strings, so missing values become "".
        cleaned = ["" if s is None or (isinstance(s, float) and s != s) else str(s) for s in sentences]
        return self.model.encode(cleaned, convert_to_tensor=True)
```

## 5.3 `src/wtl/data.py`

Three steps: guess each column's semantic type, turn every table into tensors and connect rows by key, then sample time-respecting neighborhoods.

```python
"""Load a RelBench database and task, and turn it into a PyG heterogeneous temporal graph.

Graph layout (Relational Deep Learning):
  * one node type per table, one node per row; row features live in data[table].tf (a TensorFrame)
  * one edge type per foreign key, in both directions: (child, "f2p_<fkey>", parent) and its "rev_" reverse
  * data[table].time holds each row's timestamp (Unix seconds) wherever the table has a time column
"""
import json
from pathlib import Path

from relbench import load_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader

from wtl.text import GloveTextEmbedding

# Our cache for column types and materialized tensors (raw RelBench data lives in ~/.cache/huggingface).
CACHE_ROOT = Path.home() / ".cache" / "wtl"


def load(dataset_name: str, task_name: str):
    """Download (first time only) and return the dataset, the task, and the database."""
    dataset = load_dataset(dataset_name)  # RelBench 3.x: fetched from the Hugging Face Hub
    task = dataset.load_task(task_name)   # task tables: (entity id, prediction time, label) per split
    db = dataset.get_db()                 # all tables, cut at the test timestamp (no future rows)
    return dataset, task, db


def column_stypes(db, dataset_name: str) -> dict:
    """Semantic type of every column (numerical, categorical, text, timestamp, ...), cached as JSON."""
    path = CACHE_ROOT / dataset_name / "stypes.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {table: {col: stype(s) for col, s in cols.items()} for table, cols in raw.items()}
    proposal = get_stype_proposal(db)     # RelBench's guess from each column's dtype and contents
    path.parent.mkdir(parents=True, exist_ok=True)
    # stype is a plain Enum: save .value ("numerical"), not str() ("stype.numerical"), or reloading breaks.
    serializable = {table: {col: st.value for col, st in cols.items()} for table, cols in proposal.items()}
    path.write_text(json.dumps(serializable, indent=2))
    return proposal


def build_graph(db, dataset_name: str, task, col_to_stype: dict):
    """Turn every table into tensors and connect rows through their primary/foreign keys."""
    data, col_stats = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype,
        # Text columns go through GloVe on the CPU, 256 strings at a time.
        text_embedder_cfg=TextEmbedderConfig(text_embedder=GloveTextEmbedding(device="cpu"), batch_size=256),
        # Tensors are cached per dataset, so the slow step runs once.
        cache_dir=str(CACHE_ROOT / dataset_name / "materialized"),
        # Columns that would leak this task's label are dropped (after the cache, so one cache serves all tasks).
        remove_columns=task.hidden_columns(),
    )
    return data, col_stats  # col_stats: per-column statistics the encoders use for normalization


def make_loader(data, task, split: str, num_neighbors=(128, 64), batch_size: int = 512, shuffle: bool = False):
    """Mini-batches of targets, each with its own sampled 2-hop neighborhood from before its prediction time."""
    table = task.get_table(split)         # keep this table: get_table is not cached in RelBench 3.x
    table_input = get_node_train_table_input(table=table, task=task)
    loader = NeighborLoader(
        data,
        num_neighbors=list(num_neighbors),  # up to 128 neighbors per edge type at hop 1, 64 at hop 2
        time_attr="time",                   # only rows with time <= the target's prediction time
        input_nodes=table_input.nodes,      # (entity table name, row indices of the targets)
        input_time=table_input.time,        # each target's prediction time
        transform=table_input.transform,    # attaches labels as batch[entity_table].y
        batch_size=batch_size,
        temporal_strategy="uniform",        # sample uniformly among eligible past neighbors
        shuffle=shuffle,
        num_workers=0,
    )
    return loader, table
```

## 5.4 `src/wtl/model.py` (base version)

P6 adds partial observation to this file; for now it's the standard RelBench design.

```python
"""Relational deep learning model: per-table encoders -> time encoding -> hetero GraphSAGE -> MLP head."""
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
        # Linear readout to one logit per target.
        self.head = MLP(channels, out_channels=1, norm="batch_norm", num_layers=1)

    def forward(self, batch, entity_table: str) -> torch.Tensor:
        seed_time = batch[entity_table].seed_time                 # prediction time of each target
        x_dict = self.encoder(batch.tf_dict)                      # {table: [rows in batch, channels]}
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, t in rel_time.items():
            x_dict[node_type] = x_dict[node_type] + t            # add recency to each row's vector
        x_dict = self.gnn(x_dict, batch.edge_index_dict)          # 2 rounds of message passing
        # The first len(seed_time) rows of the entity table are the targets themselves.
        return self.head(x_dict[entity_table][: seed_time.size(0)]).view(-1)
```

## 5.5 `scripts/inspect_graph.py`

```python
"""P5: build the relational graph for a task, sample one mini-batch, and run the model on it once."""
import argparse

import torch

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.model import RDLModel

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

dataset, task, db = load(args.dataset, args.task)
data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
print(data)  # every node type (table) with its row count, and every edge type (foreign key, both directions)

# One mini-batch: a few targets, each with its own time-respecting 2-hop neighborhood.
loader, _ = make_loader(data, task, "val", batch_size=args.batch_size)
batch = next(iter(loader))
num_targets = batch[task.entity_table].seed_time.size(0)
print(f"\nOne mini-batch of {num_targets} targets ({task.entity_table}):")
for node_type in batch.node_types:
    sampled = batch[node_type].num_nodes
    if sampled:
        distinct = batch[node_type].n_id.unique().numel()   # n_id = original row id in the full table
        print(f"  {node_type:22s} {sampled:6d} sampled rows, {distinct:6d} distinct, "
              f"{sampled - distinct:6d} duplicates (same row sampled for several targets)")

model = RDLModel(data, col_stats)
print(f"\nmodel parameters: {sum(p.numel() for p in model.parameters()):,}")
model.eval()
with torch.no_grad():
    logits = model(batch, task.entity_table)
print("scores (probabilities) for the targets:", torch.sigmoid(logits).numpy().round(3))
```

```powershell
python scripts\inspect_graph.py
```

The first run takes ~2.5 min: it downloads GloVe (~480 MB) and materializes every table into `~\.cache\wtl\rel-f1`. Later runs take ~25 s.

**PASS** (tested Oct 17). A `HeteroData(...)` printout with nine node types and 26 edge types, starting

```text
HeteroData(
  races={
    tf=TensorFrame([820, 5]),
    time=[820],
  },
  ...
  (races, f2p_circuitId, circuits)={ edge_index=[2, 820] },
  (circuits, rev_f2p_circuitId, races)={ edge_index=[2, 820] },
  ...
```

then

```text
One mini-batch of 8 targets (drivers):
  races                     631 sampled rows,    178 distinct,    453 duplicates (same row sampled for several targets)
  qualifying                343 sampled rows,    343 distinct,      0 duplicates (same row sampled for several targets)
  standings                 564 sampled rows,    564 distinct,      0 duplicates (same row sampled for several targets)
  constructors               19 sampled rows,     10 distinct,      9 duplicates (same row sampled for several targets)
  drivers                     8 sampled rows,      8 distinct,      0 duplicates (same row sampled for several targets)
  results                   580 sampled rows,    580 distinct,      0 duplicates (same row sampled for several targets)

model parameters: 5,073,793
scores (probabilities) for the targets: [0.563 0.576 0.532 0.605 0.603 0.561 0.579 0.614]
```

The sampled-row counts vary by a few rows between runs (three test runs gave 629, 631, and 632 race rows), because the sampler isn't seeded. The scores change every run: the model is untrained and randomly initialized.

**STOP:**

- `pyg-lib` or `NeighborLoader` errors about temporal sampling: rerun `python scripts\check_env.py` and check `pyg-lib loaded: True` (P0).
- A traceback during GloVe download: rerun; the download resumes.
- `MemoryError`: close other programs (browsers especially) and rerun.

## 5.6 Read what you printed

**The graph is cut at the test date.** `races` has 820 nodes, not the 1,101 rows on P3's card, because `get_db()` drops every row dated after the test cutoff (2010-01-01). The model can never see the test period's rows.

**Only 8 driver rows in the batch.** Two hops from a driver reach results, standings, qualifying (hop 1), then races and constructors (hop 2). Other drivers are three hops away, so the only drivers in the batch are the 8 targets.

**The key point: sampling is disjoint.** With a time attribute, `NeighborLoader` samples each target's neighborhood separately, as if each target had its own private copy of the database up to its own prediction time. So a race that 8 drivers all took part in appears up to 8 times: 631 sampled race rows were only 178 distinct races, and 19 constructor rows only 10 distinct constructors.

That matters for the whole project. "Fetch an entity once, and every prospect connected to it sees it" can't be keyed by a row's position in the batch, because the same entity sits at several positions. It must be keyed by the row's **original id**, `n_id`. P6's `reveal.py` does exactly that.

**Results, standings, and qualifying have no duplicates** because each such row belongs to exactly one driver. The shared rows are the *parents*: races and constructors. In a CRM, those are the accounts and events.

## 5.7 What fits on the laptop

The laptop has 15.6 GB of RAM. `rel-event` (30M friendship rows) and `rel-avito` (9M search-result rows, 6M ads with text) are too big to build as full graphs here. They belong to the cloud build track ([B1](B1-cloud-gpu-and-baseline.md)), after the proposal. `rel-f1` is the laptop model dataset for P6–P7.

## 5.8 Do: write proposal §3, first paragraph

In `proposal/notes.md`, 4–6 sentences:

1. The model: RDL with per-table encoders, temporal encoding, two-layer heterogeneous GraphSAGE with sum across edge types, and an MLP head. Cite Fey et al. 2024, Robinson et al. 2024, Hamilton et al. 2017, and PyG.
2. Temporal neighbor sampling (fanout 128 / 64; rows dated at or before each target's prediction time).
3. The model size: 5.07M parameters on `rel-f1`, of which 3.26M are the per-table encoders and 1.72M the GNN.
4. One sentence previewing your change (partial observation, P6).

`proposal/refs.bib` already has the model's citations under these keys: `fey2024rdl` (RDL), `hamilton2017sage` (GraphSAGE), and `fey2019pyg` (PyG). Open the file, find each one, and compare it with its PMLR or arXiv page.

The text encoder isn't in it yet. Add this entry if your write-up mentions GloVe embeddings:

```bibtex
@inproceedings{pennington2014glove,
  title     = {{GloVe}: Global Vectors for Word Representation},
  author    = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher D.},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2014}
}
```

**Self-check**

1. Why must "reveal this row" be keyed by `n_id` rather than by the row's position in the batch?
2. `drivers`, `constructors`, and `circuits` have no time column. What does the temporal encoder add to their rows?
3. What would go wrong if the loader ignored `time_attr`?

**Next: [P6 · Describe the model with figures and equations](P6-describe-the-model.md)**
