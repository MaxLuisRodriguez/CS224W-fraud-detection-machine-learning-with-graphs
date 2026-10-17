# Module 1 · Orientation and setup

> ## Project type: **1 · Real-world application of GNNs**
>
> Deliverables: a **Medium blog post + a Google Colab**. This is **not** type 2 (Tutorial on PyG functionality), and **not** type 3 (Implementation of cutting-edge research as a PyG pull request).

**Why type 1.** The handout asks type-1 projects for four things:

- a use case "motivated by real-world problems, e.g., an application that a company would actually use";
- an appropriate **public** dataset;
- a clear graph ML formulation;
- "some form of **novelty** … or **comprehensive analyses**."

This project has all four:

- **Use case:** research budgets for sales call lists, the problem Nooks' AI research and dialer products sit on.
- **Data:** public RelBench databases.
- **Formulation:** budgeted retrieval on a relational graph.
- **Novelty and analyses:** mask-aware relational GNNs plus batch-aware lookup policies (novelty), and budget frontiers, ablations, architecture comparisons, and subset analyses (analyses).

Type 3 would mean re-implementing someone else's paper as a PyG PR, and type 2 would be a library tutorial. Neither is this project.

**By the end of this module you will have:**

- [ ] A clear picture of the problem, the research question, and five testable hypotheses (Parts A–B)
- [ ] The repo skeleton created and committed (Part C)
- [ ] A local Python environment that runs the full PyG + RelBench stack on your laptop (Part C)
- [ ] Your first real finding about the data: how much prospects on a call list share (Part C)
- [ ] Every proposal requirement mapped to an answer, plus a 2-page outline (Part D)

Conventions: `PowerShell` blocks run on your laptop from the repo root. You type every line. Each step ends with **PASS** (what you should see) or **STOP** (paste the output to me).

---

## Deadlines (CS224W Fall 2026 handout, §4.1)

| Deliverable | Due (Pacific) | Weight | What to submit |
|---|---|---|---|
| **Proposal** | **Tue Oct 20, 2026, 11:59pm** | 13.33% of project, 4% of course | Group members + proposal; NeurIPS'22 style; aim for **2 pages excluding references**; **no abstract** |
| Milestone | Tue Nov 10, 2026, 11:59pm | 6.67% of project, credit/no credit | Code (data processing, training/eval) and/or report draft |
| Final | Thu Dec 10, 2026, 11:59pm | 80% of project, 24% of course | Private Medium draft link + Medium account name + Colab |

> **Action today:** the handout says solo projects are "discouraged; please see Ed." Post on Ed (or ask staff) to confirm a solo project is fine **before** Oct 20.

---

## Part A · The problem space

### A1. The sales problem, in plain words

A sales rep starts the day with a **call list**. On a parallel dialer most dials never connect, but when one does, the rep needs context on screen immediately. A **research agent** (something like SCOUT) can prepare that context by making **lookups**: pulling the prospect's recent activity, enriching their company, checking intent signals.

Lookups are not free. Enrichment providers bill per record, LLM agents bill per token, and every lookup adds latency. Nobody researches 500 prospects deeply, so the agent must **choose what to look up, for whom**.

### A2. The graph view

A CRM, like any business database, is a set of linked tables. **Relational Deep Learning** (RDL; Fey et al., ICML 2024) turns a database into a graph:

- every **row** becomes a **node** (one node type per table);
- every **foreign key** becomes an **edge** (both directions);
- every row keeps its **timestamp**, and the model may only use rows dated up to the prediction time, so there is no peeking into the future.

A GNN predicts something about a target row (here, a prospect) by passing messages from the rows around it. Two layers of message passing see two hops: the prospect's own activity, and what that activity points to.

### A3. Two kinds of lookup

| | **Pull history** `list(prospect, table)` | **Enrich** `get(entity)` |
|---|---|---|
| Returns | The prospect's recent rows in one table (searches, visits, phone requests) | One entity's own row (an ad's title and price, an event's details) |
| Reveals | New rows, so new connections | Attributes of something already known to exist |
| Sales analogue | "Show me this contact's recent activity" | "Tell me about this company" |
| Who benefits | Mostly that prospect | **Every** prospect connected to that entity |

Before an entity is enriched, it is a **stub**: the model knows it exists and who touched it, but not what it is. In the model, a stub gets a learned `[MASK]` vector.

### A4. Why plan for the whole list at once

Two things couple the prospects on a list, so planning one prospect at a time leaves value on the table:

1. **Shared entities.** If five prospects touched the same event or company, one enrichment informs all five.
2. **A shared ranking cutoff.** The rep calls the top *K*. A lookup about a prospect only matters if it can move them across that cutoff, and the cutoff depends on everyone else's scores. This coupling exists **even when prospects share nothing.**

### A5. What the data already says (your first finding)

You'll reproduce these numbers yourself in step C8. For 200 random call lists of 128 prospects each, drawn from each task's validation day:

| Dataset | Entity | Shared by 2+ prospects | Lookups a shared cache saves |
|---|---|---|---|
| `rel-avito` (ads marketplace) | ad | **0.1%** | 0.1% |
| | category | 88% | 87% |
| | location | 28% | 43% |
| `rel-event` (event invitations) | event | **34%** | **38%** |
| | friend | 10% | 10% |

Prospects with no history at all (cold starts): **17%** on `rel-avito`, **2%** on `rel-event`.

On `rel-avito`, ads are almost never shared, and the coarse categories that are shared cost almost nothing to look up. Even lists built from a single location only reach 3.5% shared ads. On `rel-event`, a third of events are shared. This gives the project a natural two-dataset design:

| Dataset · task | Sharing | Graph headroom (GNN vs. LightGBM, test AUROC) | What it tests |
|---|---|---|---|
| `rel-event` · `user-ignore` | Dense | 0.804 vs. 0.773 | Does crediting one lookup to every prospect it informs pay off? (coupling 1) |
| `rel-avito` · `user-clicks` | Sparse | 0.666 vs. 0.548 | Which prospects and which history to pull, under a ranking cutoff (coupling 2) |
| `rel-f1` · `driver-top3` | — | 0.680 vs. 0.649 | Debugging only: tiny, trains in minutes |

### A6. Vocabulary

| Term | Meaning here |
|---|---|
| Target / prospect | A row of the entity table we score (a user) |
| Call list *Q* | *N* targets scored at the same time; default *N* = 128 |
| Lookup | One costed action: a pull or an enrichment |
| Budget *B* | Total lookups for the whole list; reported as lookups per prospect |
| Stub | An entity known to exist but not yet enriched |
| Revealed set *O* | Everything fetched so far; the model sees only *O* |
| Policy π | The rule that picks the next lookup |
| Frontier | Ranking quality vs. budget, one curve per policy |
| Full information | Every lookup made: the standard RelBench GNN, our upper bound |
| Oracle-greedy | Picks lookups using the true labels; a diagnostic ceiling, never deployable |

---

## Part B · Research question and early hypotheses

**Research question.** Given a call list of *N* prospects and a budget of *B* lookups, which lookups should a research agent make so that a graph neural network ranks the list best?

**Sub-questions**

- **SQ1 Frontier:** how much ranking quality does each lookup buy, from zero lookups to full information?
- **SQ2 Training:** does a GNN need to be *trained* on partial neighborhoods to work well on them?
- **SQ3 Which lookups:** which tables and relations are worth fetching first?
- **SQ4 Joint planning:** when does planning for the whole list beat researching each prospect separately?
- **SQ5 Learning:** can a learned value model choose lookups better than strong heuristics?

**Early hypotheses.** Each is falsifiable. These are predictions to test, not results.

| # | Hypothesis | Test | Refuted if |
|---|---|---|---|
| H1 | The frontier is **concave**: most of the gap between zero lookups and full information is recovered early (≥ 50% of the AUROC gap with ≤ 25% of available lookups). | Frontier of the best policy on both datasets | The curve is roughly linear or convex |
| H2 | A GNN trained on full neighborhoods **degrades** at low budgets; training with random revelation and `[MASK]` stubs closes most of that gap. | Mask-naive vs. mask-aware model at budgets 0–full: AUROC and Brier score | No difference at low budgets |
| H3 | On `rel-avito/user-clicks`, pulling **past search results** (`SearchStream`, which holds past clicks) is the most valuable lookup type per call, because the label is future clicks. | Value per lookup by table, from oracle-greedy and policy logs | Another table dominates |
| H4 | Batch-aware policies beat case-local ones by **more on `rel-event`** (34% of events shared) **than on `rel-avito`** (0.1% of ads shared). | Same policies, both datasets; also natural vs. location-based lists | The advantage is equal, or larger on `rel-avito` |
| H5 | The learned batch-value policy matches or beats the best heuristic and a Zhou-style (2026) singleton scorer at every budget, with the largest margin at low budgets. | Frontier comparison, 5 seeds, bootstrap CIs over lists | It loses to the best heuristic. That's still a publishable C1 finding, not a dead project |

---

## Part C · Codebase and environment (laptop)

### C1. Where everything will live

```text
.
├── pyproject.toml          # makes src/wtl an installable package named "wtl"
├── src/wtl/                # the library: data, model, masks, policies (built in Modules 2–4)
├── scripts/                # runnable entry points: check_env, peek_data, sharing_stats, train_full, ...
├── notebooks/              # scratch exploration; the final Colab is assembled from src/ + scripts/
├── proposal/               # LaTeX source and figures for the Oct 20 proposal
├── tests/                  # small correctness tests (no label leakage, cutoffs respected)
├── infra/aws/              # AWS policy JSON files (Module 2)
├── outputs/                # results and checkpoints (Git-ignored; synced to S3)
└── docs/                   # plan, modules, related work
```

### C2. Commit the plan before adding code

Nothing has been committed since the initial commit. Commit the docs on their own so your history reads cleanly.

```powershell
git status                 # expect: README.md, .gitignore, docs/ (no .pem or *.local.json files)
git add README.md .gitignore docs
git commit -m "Pivot to Worth the Lookup: plan, related work, modules; archive AML plans"
git push
```

**PASS:** `git status` shows a clean tree, and the commit is on GitHub.

### C3. Create the skeleton

```powershell
# Folders for code, scripts, notebooks, proposal, tests, and AWS files.
New-Item -ItemType Directory -Force src\wtl, scripts, notebooks, proposal\figures, tests, infra\aws | Out-Null

# Git doesn't track empty folders; a .gitkeep file keeps them in the repo until they have content.
foreach ($d in "notebooks", "proposal\figures", "tests", "infra\aws") { New-Item -ItemType File -Force "$d\.gitkeep" | Out-Null }
```

Create `pyproject.toml` at the repo root. It tells Python that `src/wtl/` is a package. Libraries are installed with explicit commands in C4 instead of being listed here, because PyTorch and `pyg-lib` come from special download indexes that a `pyproject.toml` can't pin cleanly.

```toml
[project]
name = "wtl"
version = "0.1.0"
description = "Worth the Lookup: budgeted relational retrieval for call-list prioritization"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

Create `src/wtl/__init__.py` with one line:

```python
"""Worth the Lookup: budgeted relational retrieval for call-list prioritization."""
```

### C4. A local environment that runs the whole stack

Your laptop has an ARM processor, and `pyg-lib` (PyG's compiled sampler, required for time-aware sampling) has no Windows-ARM build. Windows 11 runs **x64** programs on ARM through emulation, and `pyg-lib` does ship x64 Windows builds. So the environment uses an x64 Python. It's a bit slower (the first import takes ~20 s) but complete, and it matches Colab's versions. Every command below was tested on this laptop on Oct 16.

```powershell
# 1) A virtual environment in .venv using x64 Python 3.12 (uv downloads it once, ~30 MB).
uv venv --python cpython-3.12-windows-x86_64-none .venv

# 2) Activate it for this window. Your RemoteSigned policy already allows local scripts.
.\.venv\Scripts\Activate.ps1

# 3) PyTorch first, CPU build, pinned to 2.9.1: the version RelBench's own environment is tested with.
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 4) RelBench 3.0.2 + its modeling extras (PyG, pytorch-frame, sentence-transformers, LightGBM, ...).
#    Never install the PyPI package "torch-frame"; the right one is "pytorch-frame" (pulled in here).
uv pip install "relbench[example]==3.0.2"

# 5) pyg-lib built for torch 2.9 on CPU, from PyG's own wheel index (it isn't on PyPI).
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.9.0+cpu.html

# 6) Plotting and Jupyter support for exploration and proposal figures.
uv pip install matplotlib ipykernel

# 7) Your own package, editable: edits under src/wtl take effect without reinstalling.
uv pip install -e .
```

Optional, to silence a harmless Hugging Face warning about symlinks on Windows:

```powershell
[Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_SYMLINKS_WARNING", "1", "User")  # new windows pick it up
```

In VS Code, run **Ctrl+Shift+P → Python: Select Interpreter → `.venv\Scripts\python.exe`**. New terminals will then activate the environment automatically.

> **Path-length note:** some installed files have long names. This repo's path keeps every file under Windows' 260-character limit (longest: 208). Don't move the repo into a deeply nested folder.

### C5. `scripts/check_env.py`: confirm the stack

```python
"""Environment check: confirm the Python build, the CPU/GPU stack, and every library the project needs.

Run it on the laptop (Module 1) and again on the GPU instance (Module 2).
"""
import sysconfig
from importlib.metadata import version  # reads installed versions without importing the packages

import torch
import torch_geometric.typing as pyg_typing  # flags for which compiled PyG extensions loaded

# Platform of the Python build itself: "win-amd64" on the laptop (x64 Python, even on an ARM chip),
# "linux-x86_64" on the AWS instance.
print("python build:", sysconfig.get_platform())

# Torch build, and whether it can see a GPU (False on the laptop is expected).
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# The critical flag: pyg-lib loaded => time-aware neighbor sampling works.
print("torch_geometric", version("torch_geometric"), "| pyg-lib loaded:", pyg_typing.WITH_PYG_LIB)

# The relational stack: pytorch-frame encodes table columns; relbench supplies data and tasks.
print("pytorch-frame", version("pytorch-frame"), "| relbench", version("relbench"))
```

```powershell
python scripts\check_env.py
```

**PASS:**

```text
python build: win-amd64
torch 2.9.1+cpu | CUDA available: False
torch_geometric 2.8.1 | pyg-lib loaded: True
pytorch-frame 0.3.0 | relbench 3.0.2
```

**STOP:**

- `python build: win-arm64`: the venv was made with ARM Python. Delete `.venv` and redo C4 step 1.
- `pyg-lib loaded: False`: paste the step-5 output.

### C6. `scripts/peek_data.py`: look at a database before modeling it

```python
"""Module 1: first look at a RelBench database and task (no GPU, no graph code yet)."""
import argparse

from relbench import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
args = parser.parse_args()

dataset = load_dataset(args.dataset)   # downloads from Hugging Face the first time, then reads the local cache
db = dataset.get_db()                  # every table, with rows after the test cutoff removed
print(f"{args.dataset}: val cutoff {dataset.val_timestamp.date()}, test cutoff {dataset.test_timestamp.date()}")

# Schema: size, primary key, time column, and foreign keys (each foreign key becomes a graph edge type).
for name, table in db.table_dict.items():
    print(f"  {name:20s} {len(table.df):>10,} rows  pkey={table.pkey_col}  time={table.time_col}  "
          f"fkeys={table.fkey_col_to_pkey_table}")

# Task: which entity is scored, what label, how far ahead, and how balanced the classes are.
task = dataset.load_task(args.task)
print(f"\n{args.task}: predict {task.target_col} for {task.entity_table}.{task.entity_col}, {task.timedelta} ahead")
for split in ("train", "val"):         # never look at test during development
    df = task.get_table(split).df
    print(f"  {split:5s} {len(df):>7,} rows | {df[task.time_col].nunique()} prediction dates | "
          f"positive rate {df[task.target_col].mean():.3f}")
```

```powershell
python scripts\peek_data.py --dataset rel-f1 --task driver-top3
```

**PASS** (from the Oct 16 test run): nine tables, ending with

```text
driver-top3: predict qualifying for drivers.driverId, 30 days 00:00:00 ahead
  train   1,353 rows | 59 prediction dates | positive rate 0.171
  val       588 rows | 27 prediction dates | positive rate 0.202
```

Then run `--dataset rel-event --task user-ignore` (119 MB download) and read the schema the same way.

### C7. Read what you just printed

Before moving on, answer these from the output (they belong in the proposal's dataset section):

1. Which table is the entity table, and which tables point to it?
2. Which tables have a time column, and which don't?
3. How many prediction dates does the validation split have, and what is its positive rate?

### C8. `scripts/sharing_stats.py`: your first finding

This measures the premise behind joint lookups: how often do prospects on the same call list touch the same entities? It uses DuckDB to query the parquet files directly, so it never loads the whole database into memory.

```python
"""Module 1: do prospects on a call list share entities? This is the premise behind planning lookups jointly.

For random call lists of N users from a task's validation day, count every entity (ad, event, friend, ...)
the users touched up to that day, and how many list members touched each one.
"""
import argparse

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

# Per dataset: the task whose validation day defines the call lists, the files to download,
# and a SQL query listing (UserID, kind, entity) for every entity a user touched up to $cutoff.
# {db} is filled in with the local folder holding that dataset's parquet tables.
CONFIG = {
    "rel-avito": {
        "task": "user-clicks",
        "user_col": "UserID",
        "files": ["db/VisitStream.parquet", "db/PhoneRequestsStream.parquet", "db/SearchInfo.parquet"],
        "touches": """
            SELECT UserID, 'ad' AS kind, AdID AS entity
              FROM '{db}/VisitStream.parquet' WHERE ViewDate <= $cutoff AND AdID IS NOT NULL
            UNION
            SELECT UserID, 'ad', AdID
              FROM '{db}/PhoneRequestsStream.parquet' WHERE PhoneRequestDate <= $cutoff AND AdID IS NOT NULL
            UNION
            SELECT UserID, 'category', CategoryID
              FROM '{db}/SearchInfo.parquet' WHERE SearchDate <= $cutoff AND CategoryID IS NOT NULL
            UNION
            SELECT UserID, 'location', LocationID
              FROM '{db}/SearchInfo.parquet' WHERE SearchDate <= $cutoff AND LocationID IS NOT NULL
        """,
    },
    "rel-event": {
        "task": "user-ignore",
        "user_col": "user",
        "files": ["db/event_attendees.parquet", "db/event_interest.parquet", "db/user_friends.parquet"],
        # user_friends has no time column in RelBench, so friendships count as always known.
        "touches": """
            SELECT user_id AS UserID, 'event' AS kind, event AS entity
              FROM '{db}/event_attendees.parquet' WHERE start_time <= $cutoff AND event IS NOT NULL
            UNION
            SELECT "user", 'event', event
              FROM '{db}/event_interest.parquet' WHERE "timestamp" <= $cutoff AND event IS NOT NULL
            UNION
            SELECT "user", 'friend', friend
              FROM '{db}/user_friends.parquet' WHERE friend IS NOT NULL
        """,
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-avito", choices=sorted(CONFIG))
parser.add_argument("--list_size", type=int, default=128)  # N prospects per call list
parser.add_argument("--num_lists", type=int, default=200)  # random lists to average over
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
cfg = CONFIG[args.dataset]

# Download only the files we need into the same Hugging Face cache RelBench uses.
val_file = f"{args.dataset}/tasks/{cfg['task']}/val.parquet"
root = snapshot_download(
    repo_id="stanford-star/relbench-v1",
    repo_type="dataset",
    allow_patterns=[f"{args.dataset}/{f}" for f in cfg["files"]] + [val_file],
)
db = f"{root}/{args.dataset}/db".replace("\\", "/")  # DuckDB wants forward slashes, even on Windows

# The validation day: every user scored that day is a candidate for that day's call list.
val = pd.read_parquet(f"{root}/{val_file}")
cutoff = val["timestamp"].min().to_pydatetime()

# Build call lists: each one is N distinct users drawn at random from that day.
rng = np.random.default_rng(args.seed)
pool = val[cfg["user_col"]].unique()
lists = pd.DataFrame(
    [(i, int(u)) for i in range(args.num_lists) for u in rng.choice(pool, size=args.list_size, replace=False)],
    columns=["list_id", "UserID"],
)

con = duckdb.connect()
con.register("lists", lists)

# Materialize every (user, entity) touch once. UNION (not UNION ALL) drops repeat touches.
con.execute("CREATE TEMP TABLE touches AS " + cfg["touches"].format(db=db), {"cutoff": cutoff})

# One row per (list, entity): how many prospects on that list touched it.
per_entity = con.execute("""
    SELECT l.list_id, t.kind, t.entity, COUNT(DISTINCT t.UserID) AS n_users
    FROM lists l JOIN touches t ON l.UserID = t.UserID
    GROUP BY l.list_id, t.kind, t.entity
""").df()

# Per list and kind:
#   per_prospect   = enrichments needed if every prospect is researched separately (no shared cache)
#   distinct       = enrichments needed with one shared cache for the whole list
#   shared         = fraction of distinct entities touched by 2+ prospects on the list
#   saved_by_cache = 1 - distinct / per_prospect
summary = (
    per_entity.assign(shared=per_entity.n_users >= 2)
    .groupby(["list_id", "kind"])
    .agg(distinct=("entity", "size"), per_prospect=("n_users", "sum"), shared=("shared", "mean"))
    .reset_index()
)
summary["saved_by_cache"] = 1 - summary.distinct / summary.per_prospect

# Cold starts: prospects with no history at all. No lookup can reveal anything about them.
cold = con.execute("""
    SELECT AVG(CASE WHEN t.UserID IS NULL THEN 1.0 ELSE 0.0 END)
    FROM lists l LEFT JOIN (SELECT DISTINCT UserID FROM touches) t ON l.UserID = t.UserID
""").fetchone()[0]

print(f"{args.dataset}: {args.num_lists} random call lists of {args.list_size} prospects, day {cutoff.date()}\n")
print(summary.groupby("kind")[["distinct", "per_prospect", "shared", "saved_by_cache"]].mean().round(3))
print(f"\nprospects with no history (cold starts): {cold:.1%}")
```

```powershell
python scripts\sharing_stats.py --dataset rel-avito    # ~125 MB download the first time
python scripts\sharing_stats.py --dataset rel-event    # ~120 MB
```

**PASS** (Oct 16 test run, seed 0). `rel-avito`:

```text
          distinct  per_prospect  shared  saved_by_cache
ad        10220.555     10234.100   0.001           0.001
category     36.220       283.490   0.882           0.871
location    105.410       184.875   0.279           0.430
prospects with no history (cold starts): 17.0%
```

`rel-event`:

```text
        distinct  per_prospect  shared  saved_by_cache
event   1176.645      1963.490   0.343           0.382
friend  1402.250      1568.475   0.099           0.104
prospects with no history (cold starts): 2.2%
```

Try `--list_size 32` and `--list_size 512`. Sharing grows with list size, and that becomes one of your analyses.

### C9. Commit

```powershell
git add pyproject.toml src scripts notebooks proposal tests infra
git status      # no .venv, outputs, data, or *.local.json should appear (all Git-ignored)
git commit -m "Module 1: package skeleton, environment check, data peek, sharing statistics"
git push
```

---

## Part D · Proposal requirements and how this project meets them

From handout §1.3 (project type 1) and §4.1 (format).

| Requirement | Our answer | Evidence | Status |
|---|---|---|---|
| Who is in your group | Max Rodriguez (solo) | Confirm on Ed (solo is discouraged) | **Action** |
| Which dataset(s)? | RelBench `rel-event`/`user-ignore` and `rel-avito`/`user-clicks`; `rel-f1` for debugging | Part A5 | Ready |
| **Is the dataset public?** *(strict)* | Yes. Openly downloadable, no login, from Hugging Face [`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1) (not gated). Originally from public Kaggle competitions ([Avito Context Ad Clicks](https://www.kaggle.com/c/avito-context-ad-clicks); [Event Recommendation Engine Challenge](https://www.kaggle.com/c/event-recommendation-engine-challenge)). The Colab downloads it; nothing is re-hosted. | HF card license field is "other": cite RelBench and the original sources | Ready |
| Describe dataset / task / metric; cite | Tables and row counts; label definitions (ignore ≥3 invitations in 7 days; click >1 ad in 4 days); temporal splits; positive rates. Metrics: AUROC (RelBench official), average precision, Precision@K with K = ⌈0.1N⌉, and the frontier of AUROC vs. lookups per prospect | C6–C8 output; [brief](https://claude.ai/artifact/EsWEkYYaUxo7spH91Y6KCV) | Ready |
| Why this dataset? | Real temporal splits and a maintained leaderboard; measured graph headroom; **measured sharing** (dense on `rel-event`, sparse on `rel-avito`, a built-in contrast); clear sales analogues | Part A5 | Ready |
| Which graph ML model? (cite) | Relational Deep Learning: per-table encoders + temporal encoding + heterogeneous GraphSAGE (Fey et al. 2024; Robinson et al. 2024; Hamilton et al. 2017; PyG, Fey & Lenssen 2019). Our additions: mask-aware training and lookup policies | `PROJECT_PLAN.md` §5 | Ready |
| Describe the model with figures and equations | Fig. 1: call list as a graph with lookups. Fig. 2: database schema. Equations below | Figures in the brief; LaTeX below | Write this weekend |
| Why is the model appropriate? | Data is relational, so message passing over foreign keys uses it directly. Temporal sampling prevents leakage. Shared entities are literally shared nodes, so one enrichment updates every prospect connected to it. `[MASK]` stubs represent "known to exist, not yet fetched" | Parts A2–A4 | Ready |
| Novelty, or comprehensive analyses (type-1 bar, §1) | Both. New formulation (budgeted retrieval for a target batch on relational graphs), mask-aware RDL, batch-aware lookup value; plus frontiers, ablations (mask vs. no mask, batch vs. per-prospect credit), architectures (GraphSAGE vs. GAT vs. LightGBM), subsets (cold start, sharing level), explainability (which tables are worth fetching) | `PROJECT_PLAN.md` §6–7; `RELATED_WORK.md` | Ready |
| Format | NeurIPS'22 style (`\usepackage[final]{neurips_2022}` so your name shows), ~2 pages without references, no abstract | [style files](https://neurips.cc/Conferences/2022/PaperInformation/StyleFiles) | Set up below |
| Preliminary results *(not required; strengthens it)* | Sharing statistics (done); full-information baseline numbers (Module 2) | C8; Module 2 | Weekend |

### D1. Two-page outline

1. **Application and motivation** (~⅓ page). Call lists, the cost of research, and the question. One sentence on why GNNs.
2. **Dataset, task, metric** (~½ page). A small table with both datasets (rows, label, horizon, splits, positive rate, GNN vs. LightGBM), the sharing statistic, public availability, citations.
3. **Graph ML approach** (~¾ page).
   - Fig. 1, the call-list graph.
   - The RDL model, the masking change, the objective, and the lookup value (equations below).
   - Why it fits.
   - Baselines: random, case-local round-robin, uncertainty-first, coverage, Zhou-style singleton scorer, oracle-greedy, full information.
4. **Evaluation plan and preliminary results** (~⅓ page). Frontier and the analyses list, H1–H5 in one sentence each, and the sharing numbers plus baseline reproduction.
5. **References** (don't count toward the 2 pages). Use the "Read tonight" papers from the brief plus PyG and GraphSAGE.

### D2. Equations, ready for LaTeX

```latex
% Initial embedding of row v at prediction time t: table encoder + time-since encoding.
h_v^{(0)} = \mathrm{Enc}_{T(v)}(x_v) + \phi\big(t - \tau(v)\big)

% Heterogeneous GraphSAGE layer: one SAGE update per relation r, summed, then LayerNorm + ReLU;
% only neighbors u with timestamp tau(u) <= t are sampled.
h_v^{(\ell+1)} = \mathrm{ReLU}\Big(\mathrm{LN}\Big(\sum_{r \in \mathcal{R}} W_{r,1}^{(\ell)} h_v^{(\ell)}
  + W_{r,2}^{(\ell)} \!\!\sum_{u \in \mathcal{N}_r(v),\, \tau(u) \le t}\!\! h_u^{(\ell)}\Big)\Big)

% Our change: rows not yet fetched (stubs) enter as a learned per-table [MASK] vector.
\tilde h_v^{(0)} = m_v\, h_v^{(0)} + (1 - m_v)\, e^{\mathrm{mask}}_{T(v)}, \qquad m_v = \mathbb{1}[v \in O]

% Objective: choose a lookup policy that maximizes ranking quality of the call list under budget B.
\max_{\pi}\; \mathbb{E}\Big[ M\big( f_\theta(Q \mid O^{\pi}_{B}),\, y \big) \Big]
  \quad \text{s.t.}\quad |O^{\pi}_{B}| - |O_0| \le B

% Batch value of a candidate lookup a: the drop in total loss over the whole list (learned at test time).
V(a \mid O) = \sum_{i \in Q} \Big[ \ell\big(f_\theta(q_i \mid O), y_i\big) - \ell\big(f_\theta(q_i \mid O \cup \{a\}), y_i\big) \Big]
```

`HeteroGraphSAGE` in RelBench uses sum aggregation across relations and LayerNorm + ReLU after each layer, as written above. The aggregator inside each relation is a choice (`aggr="sum"` in our model).

### D3. Set up the proposal source (5 minutes)

```powershell
# Official NeurIPS 2022 style file and template, into proposal/.
Invoke-WebRequest https://media.neurips.cc/Conferences/NeurIPS2022/Styles/neurips_2022.sty -OutFile proposal\neurips_2022.sty
Invoke-WebRequest https://media.neurips.cc/Conferences/NeurIPS2022/Styles/neurips_2022.tex -OutFile proposal\template_neurips_2022.tex
```

Write `proposal/proposal.tex` from the template:

- use `\usepackage[final]{neurips_2022}` so your name appears;
- delete the abstract;
- follow the D1 outline.

Overleaf works too: upload the same two files.

---

## Part E · Check yourself, then move on

**Say it in 60 seconds** (the Nooks dinner version): *"A call list is a graph. Prospects share companies, events, and activity, so one lookup can sharpen several predictions, and some lookups only pay off after another one. Research costs money, tokens, and latency, so the question is which lookups to make for the whole list. I measure it as a curve of ranking quality per lookup, on public relational data, and I'm testing whether a learned GNN planner beats the obvious heuristics. On real data, a third of events are shared across a random 128-person list, but almost no ads are, so I can test when joint planning matters and when it doesn't."*

**Self-check**

1. Why can't a lookup help a cold-start prospect, and what fraction of `rel-avito` prospects are cold starts?
2. What is the difference between pulling history and enriching an entity, and which one benefits several prospects?
3. Name the two reasons the whole list must be planned together. Which one exists even with zero shared entities?
4. Why must the GNN only see rows dated up to the prediction time?
5. Which hypothesis would you drop first if `rel-event`'s graph headroom (0.773 → 0.804) turns out too small to show differences?

**Next: [Module 2](02-cloud-gpu-and-baseline.md)**: AWS GPU instance, the PyG graph, and the full-information baseline. That gives you the upper-bound numbers for the proposal.
