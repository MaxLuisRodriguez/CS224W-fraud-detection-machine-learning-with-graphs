"""P4: do prospects on a call list share entities? This is the premise behind planning lookups jointly.

For random call lists of N users from a task's validation day, count every entity (ad, event, friend, ...)
the users touched up to that day, and how many list members touched each one.
"""
import argparse

import duckdb                                   # SQL engine that queries parquet files in place
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

# =====================================================================================================
# What a "touch" means in each dataset
# =====================================================================================================
# For each dataset: the task whose validation day defines the call lists, the parquet files we need,
# and a SQL query listing one row (UserID, kind, entity) per entity a user touched up to $cutoff.
#   {db}      -> filled in below with the local folder holding that dataset's parquet tables
#   $cutoff   -> a query parameter; DuckDB substitutes the validation day safely at run time
#   UNION     -> stacks the results of several SELECTs AND removes duplicate rows
#                (UNION ALL would keep duplicates; we want each user-entity pair once)
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

# =====================================================================================================
# Download only the files we need
# =====================================================================================================
# snapshot_download(repo_id, repo_type, allow_patterns):
#   takes a Hugging Face repo and a list of file patterns; downloads only matching files (resuming if
#   interrupted, skipping files already cached); returns the local folder path of the cached copy.
val_file = f"{args.dataset}/tasks/{cfg['task']}/val.parquet"
root = snapshot_download(
    repo_id="stanford-star/relbench-v1",
    repo_type="dataset",
    allow_patterns=[f"{args.dataset}/{f}" for f in cfg["files"]] + [val_file],
)
db = f"{root}/{args.dataset}/db".replace("\\", "/")  # DuckDB wants forward slashes, even on Windows

# =====================================================================================================
# The validation day, and the random call lists drawn from it
# =====================================================================================================
val = pd.read_parquet(f"{root}/{val_file}")
# .min() -> the earliest timestamp (pandas Timestamp); .to_pydatetime() -> a plain Python datetime,
# the type DuckDB accepts as a query parameter.
cutoff = val["timestamp"].min().to_pydatetime()

rng = np.random.default_rng(args.seed)
pool = val[cfg["user_col"]].unique()            # .unique(): every distinct user id scored that day
# rng.choice(pool, size=N, replace=False): picks N distinct users from the pool -> one call list.
# The list comprehension repeats that num_lists times and records (list_id, UserID) pairs.
lists = pd.DataFrame(
    [(i, int(u)) for i in range(args.num_lists) for u in rng.choice(pool, size=args.list_size, replace=False)],
    columns=["list_id", "UserID"],
)

# =====================================================================================================
# Count touches with SQL
# =====================================================================================================
con = duckdb.connect()                          # an in-memory DuckDB database (nothing written to disk)
con.register("lists", lists)                    # exposes the pandas DataFrame to SQL as a table named "lists"

# con.execute(sql, params): runs SQL, filling $cutoff from the params dict.
# CREATE TEMP TABLE ... AS: saves the query result as a temporary table, so it's computed once.
con.execute("CREATE TEMP TABLE touches AS " + cfg["touches"].format(db=db), {"cutoff": cutoff})

# One row per (list, entity): how many prospects on that list touched it.
#   JOIN ... ON l.UserID = t.UserID   -> keeps only the touches of users who are on each list
#   COUNT(DISTINCT t.UserID)          -> how many different list members touched that entity
#   .df()                             -> returns the query result as a pandas DataFrame
per_entity = con.execute("""
    SELECT l.list_id, t.kind, t.entity, COUNT(DISTINCT t.UserID) AS n_users
    FROM lists l JOIN touches t ON l.UserID = t.UserID
    GROUP BY l.list_id, t.kind, t.entity
""").df()

# =====================================================================================================
# Summarize per list and kind
# =====================================================================================================
#   per_prospect   = enrichments needed if every prospect is researched separately (no shared cache)
#   distinct       = enrichments needed with one shared cache for the whole list
#   shared         = fraction of distinct entities touched by 2+ prospects on the list
#   saved_by_cache = 1 - distinct / per_prospect
# .assign(shared=...)   -> returns a copy with a new True/False column: touched by 2+ prospects?
# .groupby([...]).agg(name=(column, function)) -> one row per (list, kind) with the named summaries:
#     "size" counts rows (distinct entities), "sum" adds n_users, "mean" of True/False = fraction True
# .reset_index()        -> turns the group keys back into ordinary columns
summary = (
    per_entity.assign(shared=per_entity.n_users >= 2)
    .groupby(["list_id", "kind"])
    .agg(distinct=("entity", "size"), per_prospect=("n_users", "sum"), shared=("shared", "mean"))
    .reset_index()
)
summary["saved_by_cache"] = 1 - summary.distinct / summary.per_prospect

# Cold starts: prospects with no history at all. No lookup can reveal anything about them.
#   LEFT JOIN keeps every list member, filling NULL where the user has no touches;
#   AVG(CASE WHEN ... IS NULL THEN 1 ELSE 0) is the fraction of members with no touches.
#   .fetchone()[0] -> the single value of the single result row
cold = con.execute("""
    SELECT AVG(CASE WHEN t.UserID IS NULL THEN 1.0 ELSE 0.0 END)
    FROM lists l LEFT JOIN (SELECT DISTINCT UserID FROM touches) t ON l.UserID = t.UserID
""").fetchone()[0]

print(f"{args.dataset}: {args.num_lists} random call lists of {args.list_size} prospects, day {cutoff.date()}\n")
print(summary.groupby("kind")[["distinct", "per_prospect", "shared", "saved_by_cache"]].mean().round(3))
print(f"\nprospects with no history (cold starts): {cold:.1%}")
