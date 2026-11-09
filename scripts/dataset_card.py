"""P3: a dataset card for the proposal, built from RelBench's own manifests on Hugging Face.

Reads each table's row count from its parquet footer (a few KB), so nothing large is downloaded.

Question it answers: what is in one RelBench dataset and task, in a form ready to paste into a write-up?
It prints Markdown: a table describing the database (rows, keys and time column of every table) and a
table describing the task (rows, prediction dates and share of positives for each split).

Two terms used below:
- A manifest is a small YAML text file in the RelBench repository that describes a dataset or a task.
- Parquet is the file format RelBench stores tables in. A parquet file ends with a small "footer" of
  metadata that includes its row count, so counting rows only needs that footer, not the data.
Only the task's train and val label tables are read in full, because the positive rate needs the labels.
The test labels are never read.

How to run (from the repo root; needs internet access):
    python scripts/dataset_card.py                                     # rel-event / user-ignore
    python scripts/dataset_card.py --dataset rel-f1 --task driver-top3

A healthy run prints a heading with the dataset description, the two cutoff dates, one Markdown row per
database table, then the task description and a split table whose test row says "held out".

Where it fits: proposal step P3 (dataset, task and metric).
"""
import argparse

import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem

REPO = "datasets/stanford-star/relbench-v1"  # Hugging Face path of the RelBench v1 datasets

# Command-line options: which dataset and task to describe.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-event")
parser.add_argument("--task", default="user-ignore")
args = parser.parse_args()

# A range request asks the server for only some bytes of a file, which is how the footer-only reads stay small.
fs = HfFileSystem()  # lets us open files in the Hugging Face repo like local files (with range requests)
# Every file of one dataset lives under this folder of the repo.
base = f"{REPO}/{args.dataset}"

# The database manifest: description, split cutoffs, and every table's keys and time column.
# fs.read_text(path) fetches one small text file; yaml.safe_load turns its YAML text into Python dicts and lists.
manifest = yaml.safe_load(fs.read_text(f"{base}/manifest.yaml"))
print(f"## {args.dataset}: {manifest['description']}\n")
print(f"Validation cutoff {manifest['val_timestamp']}, test cutoff {manifest['test_timestamp']}.\n")
# Markdown table header; "---:" right-aligns the Rows column.
print("| Table | Rows | Primary key | Time column | Foreign keys |")
print("|---|---:|---|---|---|")
for name, table in manifest["tables"].items():
    # fs.open(..., "rb") opens the remote parquet file for reading as bytes, without downloading all of it.
    # pq.ParquetFile(f).metadata parses only the footer; num_rows is the row count stored there.
    with fs.open(f"{base}/db/{name}.parquet", "rb") as f:
        rows = pq.ParquetFile(f).metadata.num_rows   # read from the footer only
    # Foreign keys as "column -> parent table". The manifest stores None for a table without foreign keys,
    # so `or {}` swaps in an empty dict, and the final `or "none"` replaces an empty result with "none".
    fkeys = ", ".join(f"{col} -> {parent}" for col, parent in (table["fkeys"] or {}).items()) or "none"
    # {rows:,} prints the count with thousands separators; `or 'none'` fills a missing key or time column.
    print(f"| `{name}` | {rows:,} | {table['pkey'] or 'none'} | {table['time_col'] or 'none'} | {fkeys} |")

# The task manifest: what is predicted, for which entity, over what horizon.
task = yaml.safe_load(fs.read_text(f"{base}/tasks/{args.task}/manifest.yaml"))
print(f"\n### Task `{args.task}`: {task['description']}\n")
print(f"Type {task['task_type']}; entity `{task['entity_table']}.{task['entity_col']}`; "
      f"label `{task['target_col']}`; horizon {task['timedelta']}.\n")
print("| Split | Rows | Prediction dates | Positive rate |")
print("|---|---:|---:|---:|")
for split in ("train", "val"):
    # These label tables are read in full: pd.read_parquet returns the whole file as a pandas DataFrame.
    with fs.open(f"{base}/tasks/{args.task}/{split}.parquet", "rb") as f:
        df = pd.read_parquet(f)
    # nunique() counts distinct prediction dates; the mean of a 0/1 label column is the share of positives.
    print(f"| {split} | {len(df):,} | {df[task['time_col']].nunique()} | {df[task['target_col']].mean():.3f} |")
# The test split: row count from the footer only.
with fs.open(f"{base}/tasks/{args.task}/test.parquet", "rb") as f:
    test_rows = pq.ParquetFile(f).metadata.num_rows  # count only; test labels stay unread until the end
print(f"| test | {test_rows:,} | held out | held out |")
