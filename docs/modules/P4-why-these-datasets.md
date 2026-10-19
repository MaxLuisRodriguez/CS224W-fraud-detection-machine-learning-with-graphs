# P4 · Why these datasets? (~45 min)

> **Handout bullet (§1.3):** "Why did you choose the dataset?"
>
> **Proposal section this feeds:** §2, second half (after the dataset/task/metric description from P3).

The answer has two measured parts: the graph must help at all (headroom), and the prospects on a call list must share things a lookup can reveal (sharing). The two primary datasets differ sharply on the second, and that difference is a hypothesis test in itself (H4).

## 4.1 Criterion 1: the graph has to help

A lookup reveals relational context. If relational context doesn't beat the target's own row, no lookup policy can show anything. The [RelBench leaderboard](https://relbench.stanford.edu/) reports, per task, a GNN (RDL) and LightGBM trained on the target's own row. Test AUROC, v3 leaderboard, checked Oct 10:

| Dataset · task | GNN | LightGBM | Gap | Decision |
|---|---:|---:|---:|---|
| `rel-avito` · `user-clicks` | 0.666 | 0.548 | **+0.118** | Primary (sparse sharing) |
| `rel-event` · `user-ignore` | 0.804 | 0.773 | +0.031 | Primary (dense sharing) |
| `rel-f1` · `driver-top3` | 0.680 | 0.649 | +0.031 | Debugging and laptop results |
| `rel-event` · `user-repeat` | 0.686 | 0.689 | −0.003 | **Excluded**: no graph advantage, so nothing for lookups to recover |

These are leaderboard numbers, not yours. Cite the leaderboard, and recheck it the day you submit.

## 4.2 Criterion 2: do prospects on a call list share entities?

`scripts/sharing_stats.py` uses DuckDB to query the parquet files directly, so it never loads a whole database into memory.

```python
"""P4: do prospects on a call list share entities? This is the premise behind planning lookups jointly.

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

**PASS** (tested Oct 17, seed 0; a few seconds each once downloaded):

```text
rel-avito: 200 random call lists of 128 prospects, day 2015-05-08

           distinct  per_prospect  shared  saved_by_cache
kind
ad        10220.555     10234.100   0.001           0.001
category     36.220       283.490   0.882           0.871
location    105.410       184.875   0.279           0.430

prospects with no history (cold starts): 17.0%
```

```text
rel-event: 200 random call lists of 128 prospects, day 2012-11-21

        distinct  per_prospect  shared  saved_by_cache
kind
event   1176.645      1963.490   0.343           0.382
friend  1402.250      1568.475   0.099           0.104

prospects with no history (cold starts): 2.2%
```

**STOP:**

- `duckdb.IOException` (file not found): the download was interrupted. Rerun; `snapshot_download` resumes.
- Different numbers: check `--seed 0` and `--num_lists 200`. Anything else, paste the output.

## 4.3 Read the numbers

- **`rel-event` is dense.** 34% of the events a call list touches are shared by 2+ prospects, and a shared cache saves 38% of event lookups. Friends: 10%.
- **`rel-avito` is sparse where it counts.** Ads are what a lookup would enrich, and 0.1% of them are shared. Categories (88%) and locations (28%) are shared, but there are only 68 categories and 3,512 locations, so a handful of lookups covers them. Ads dominate the work: about 10,200 distinct ads per list against 36 categories.
- **Summed over all kinds**, from the mean counts above, a shared cache saves about **27%** of lookups on `rel-event` (1 − 2,579 / 3,532) but only about **3%** on `rel-avito` (1 − 10,362 / 10,702).
- **Cold starts** set a ceiling on what research can do: 17% of `rel-avito` prospects and 2% of `rel-event` prospects have no history, so no lookup can tell the model anything about them.
- In an Oct 16 test run, call lists drawn from a single location raised `rel-avito` ad sharing only to 3.5%. That variant isn't in this script, so cite it as a side note or leave it out.

## 4.4 Exercise: does sharing grow with list size?

Predict the answer first, then run:

```powershell
python scripts\sharing_stats.py --dataset rel-event --list_size 32
python scripts\sharing_stats.py --dataset rel-event --list_size 512
python scripts\sharing_stats.py --dataset rel-avito --list_size 32
python scripts\sharing_stats.py --dataset rel-avito --list_size 512
```

**PASS** (tested Oct 17; the 128 column is from 4.2):

| Shared by 2+ prospects (cache saves) | *N* = 32 | *N* = 128 | *N* = 512 |
|---|---:|---:|---:|
| `rel-event` events | 13.2% (12.8%) | 34.3% (38.2%) | 52.6% (66.9%) |
| `rel-event` friends | 2.9% (2.9%) | 9.9% (10.4%) | 25.9% (29.4%) |
| `rel-avito` ads | 0.0% (0.0%) | 0.1% (0.1%) | 0.5% (0.5%) |

Sharing grows with list size on `rel-event`, so joint planning should matter more for bigger lists there. On `rel-avito`, ads stay unshared even at 512. List size is one of the analyses in the final project.

## 4.5 The two-dataset design

| | `rel-event` · `user-ignore` | `rel-avito` · `user-clicks` |
|---|---|---|
| Sharing | **Dense**: one lookup often informs several prospects | **Sparse**: almost every ad lookup informs one prospect |
| Coupling it tests (P1 §1.4) | Shared entities | The shared ranking cutoff, with almost no shared entities |
| Graph headroom | Small (+0.031) | Large (+0.118) |
| Role in H4 | Batch-aware policies should win by more here | …than here |

Both datasets map onto a sales stack:

| `rel-event` table | Sales analogue | `rel-avito` table | Sales analogue |
|---|---|---|---|
| `users` | prospects | `UserInfo` | prospects |
| `event_attendees` (invited / yes / no / maybe) | outreach touches and responses | `SearchInfo` | intent: what they searched for |
| `events` | campaigns, webinars | `SearchStream` | what they were shown and clicked |
| `user_friends` | colleagues, buying-committee links | `VisitStream` | website visits |
| `event_interest` | expressed interest | `PhoneRequestsStream` | hand-raises ("give me the seller's number") |
| | | `AdsInfo`, `Category`, `Location` | products and accounts, segment, territory |

The other reasons, in one line each:

- **Public** and downloadable without login (P2).
- **Temporal splits with leakage-safe labels** built in, so no hand-rolled split can leak the future.
- **Maintained baselines** (GNN and LightGBM), so the full-information end of every frontier is a known number.
- **`rel-f1` is tiny** (1,353 training rows), so the whole pipeline trains on the laptop in minutes. The two primary datasets need the cloud (P5).

## 4.6 Do: write proposal §2, why these datasets

In `proposal/notes.md`, 4–6 sentences after the P3 material:

1. The headroom rule, with the two primary gaps and the excluded `user-repeat` task.
2. The measured sharing contrast: 34% vs. 0.1% shared, 27% vs. 3% of lookups saved by a shared cache (state *N* = 128, 200 lists, seed 0).
3. What the contrast tests (H4), in one sentence.
4. Cold starts as a ceiling on gains.
5. One sentence of sales analogues per dataset.

**Self-check**

1. Why is `rel-event` · `user-repeat` excluded even though it's from one of your primary databases?
2. Categories are 88% shared on `rel-avito`. Why doesn't that make `rel-avito` a dense-sharing dataset for this project?
3. Which prospects can no lookup help, and how common are they on each dataset?

**Next: [P5 · Which graph ML model?](P5-which-model.md)**
