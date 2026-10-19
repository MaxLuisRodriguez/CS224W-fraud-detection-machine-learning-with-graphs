"""P3: a dataset card for the proposal, built from RelBench's own manifests on Hugging Face.

Reads each table's row count from its parquet footer (a few KB), so nothing large is downloaded.
"""
import argparse

import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem

REPO = "datasets/stanford-star/relbench-v1"  # Hugging Face path of the RelBench v1 datasets

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-event")
parser.add_argument("--task", default="user-ignore")
args = parser.parse_args()

fs = HfFileSystem()  # lets us open files in the Hugging Face repo like local files (with range requests)
base = f"{REPO}/{args.dataset}"

# The database manifest: description, split cutoffs, and every table's keys and time column.
manifest = yaml.safe_load(fs.read_text(f"{base}/manifest.yaml"))
print(f"## {args.dataset}: {manifest['description']}\n")
print(f"Validation cutoff {manifest['val_timestamp']}, test cutoff {manifest['test_timestamp']}.\n")
print("| Table | Rows | Primary key | Time column | Foreign keys |")
print("|---|---:|---|---|---|")
for name, table in manifest["tables"].items():
    with fs.open(f"{base}/db/{name}.parquet", "rb") as f:
        rows = pq.ParquetFile(f).metadata.num_rows   # read from the footer only
    fkeys = ", ".join(f"{col} -> {parent}" for col, parent in (table["fkeys"] or {}).items()) or "none"
    print(f"| `{name}` | {rows:,} | {table['pkey'] or 'none'} | {table['time_col'] or 'none'} | {fkeys} |")

# The task manifest: what is predicted, for which entity, over what horizon.
task = yaml.safe_load(fs.read_text(f"{base}/tasks/{args.task}/manifest.yaml"))
print(f"\n### Task `{args.task}`: {task['description']}\n")
print(f"Type {task['task_type']}; entity `{task['entity_table']}.{task['entity_col']}`; "
      f"label `{task['target_col']}`; horizon {task['timedelta']}.\n")
print("| Split | Rows | Prediction dates | Positive rate |")
print("|---|---:|---:|---:|")
for split in ("train", "val"):
    with fs.open(f"{base}/tasks/{args.task}/{split}.parquet", "rb") as f:
        df = pd.read_parquet(f)
    print(f"| {split} | {len(df):,} | {df[task['time_col']].nunique()} | {df[task['target_col']].mean():.3f} |")
with fs.open(f"{base}/tasks/{args.task}/test.parquet", "rb") as f:
    test_rows = pq.ParquetFile(f).metadata.num_rows  # count only; test labels stay unread until the end
print(f"| test | {test_rows:,} | held out | held out |")
