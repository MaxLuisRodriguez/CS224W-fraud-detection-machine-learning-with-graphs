"""P2: load a RelBench database and task with no login, and print what's in it"""

import argparse

from relbench import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")

args = parser.parse_args()

dataset = load_dataset(args.dataset) # downloads from Hugging Face the first time, then reads the local cache
db = dataset.get_db() # every table, with rows after the test cutoff removed
print(f"{args.dataset}: val cutoff {dataset.val_timestamp.date()}, test cutoff {dataset.test_timestamp.date()}")

# Schema: size, primary key, time column, and foreign keys (each foreign key becomes a graph edge type).
for name, table in db.table_dict.items():
    print(f" {name:20s} {len(table.df):>10,} rows pkey={table.pkey_col} time={table.time_col} "
          f"fkeys={table.fkey_col_to_pkey_table}")

# Task: which entity is scored, what label, how far ahead, and class balance.
task = dataset.load_task(args.task)

print(f"\n{args.task}: predict {task.target_col} for {task.entity_table}.{task.entity_col}, {task.timedelta} ahead")

for split in ("train", "val"):
    df = task.get_table(split).df
    print(f"  {split:5s} {len(df):>7,} rows | {df[task.time_col].nunique()} prediction dates | "
          f"positive rate {df[task.target_col].mean():.3f}")
