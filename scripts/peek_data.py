"""P2: load a RelBench database and task with no login, and print what's in it.

Question it answers: which tables, keys, dates and labels does a RelBench dataset contain?
RelBench is a benchmark of real relational databases (several linked tables) with prediction tasks defined
on them. The data is public on Hugging Face, so no account or login is needed.

How to run (from the repo root):
    python scripts/peek_data.py                                        # rel-f1 / driver-top3
    python scripts/peek_data.py --dataset rel-event --task user-ignore

What it prints:
1. The validation and test cutoff dates. Training examples have prediction times before the validation
   cutoff, validation examples fall between the two cutoffs, and test examples come after the test cutoff.
2. One line per table: row count, primary key (pkey, the column that identifies a row), time column, and
   foreign keys (fkeys, as {column: table it points to}). Each foreign key becomes edges in the graph.
3. The task: which label is predicted, for which entity, and how far ahead.
4. For the train and val splits: the number of rows, the number of distinct prediction dates, and the share
   of positive labels. The test split is left alone on purpose.

A healthy run prints all of this without errors. The first run downloads the data; later runs read the
local cache.

Where it fits: proposal step P2 (datasets), the first look at the data.
"""

import argparse

from relbench import load_dataset

# Command-line options: which RelBench dataset and which of its tasks to describe.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")

args = parser.parse_args()

# load_dataset(name) returns a RelBench Dataset object, which knows its cutoff dates and its tasks.
dataset = load_dataset(args.dataset) # downloads from Hugging Face the first time, then reads the local cache
# get_db() returns a Database; db.table_dict maps each table name to a Table (a pandas DataFrame plus its keys).
db = dataset.get_db() # every table, with rows after the test cutoff removed
# .date() drops the clock time from a timestamp, leaving only the day.
print(f"{args.dataset}: val cutoff {dataset.val_timestamp.date()}, test cutoff {dataset.test_timestamp.date()}")

# Schema: size, primary key, time column, and foreign keys (each foreign key becomes a graph edge type).
# None means the table has no such column (for example a table with no dates has time=None).
for name, table in db.table_dict.items():
    print(f" {name:20s} {len(table.df):>10,} rows pkey={table.pkey_col} time={table.time_col} "
          f"fkeys={table.fkey_col_to_pkey_table}")

# Task: which entity is scored, what label, how far ahead, and class balance.
# load_task(name) returns the task: entity_table/entity_col say who is scored, target_col is the label column,
# and timedelta is the horizon (how far after the prediction time the label looks).
task = dataset.load_task(args.task)

print(f"\n{args.task}: predict {task.target_col} for {task.entity_table}.{task.entity_col}, {task.timedelta} ahead")

for split in ("train", "val"):
    # get_table(split).df: the split's task table, one row per (entity, prediction time) with its label.
    df = task.get_table(split).df
    # nunique() counts distinct values; the mean of a 0/1 label column is the share of positives.
    print(f"  {split:5s} {len(df):>7,} rows | {df[task.time_col].nunique()} prediction dates | "
          f"positive rate {df[task.target_col].mean():.3f}")
