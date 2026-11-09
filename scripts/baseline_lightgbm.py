"""P7: LightGBM on each target's own row only. This is the "zero lookups" reference point.

What this answers:
    How well can we rank prospects using ONLY the facts in each prospect's own row, with no graph and no
    lookups at all? That tells us what the own row alone is worth, so we can see how much the graph adds.

How it works:
    LightGBM is a gradient-boosted decision-tree model, a strong default for one flat table. Each example
    is one (target, prediction time, label) row of the task, joined with that target's own columns (for
    example a user's profile row on rel-event). No other table is used.

How to run (from the repo root):
    python scripts/baseline_lightgbm.py --dataset rel-event --task user-ignore

Reads the RelBench dataset and task. Writes nothing; it prints three lines.

How to read the output:
    "features used" lists the columns the trees were allowed to split on.
    "RelBench val metrics" is RelBench's own scoring over the whole validation table (includes roc_auc).
    "call lists of N" averages AUROC, average precision and precision@K over random validation lists of N
    prospects, the same call-list metrics that train_rdl.py reports. A value from here can be drawn as an
    extra reference line with fig_frontier.py --ref.
"""
import argparse

import lightgbm as lgb
import pandas as pd

from relbench import load_dataset

from wtl.metrics import call_list_metrics

# Command-line options: dataset, task, and how many prospects make up one call list for the call-list metrics.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--list_size", type=int, default=128)
args = parser.parse_args()

# Load the dataset, the task (what to predict, for which table) and the database (every table).
dataset = load_dataset(args.dataset)
task = dataset.load_task(args.task)
db = dataset.get_db()
entity = db.table_dict[task.entity_table]   # the targets' own table (drivers on rel-f1, users on rel-event)

# Join each (target, prediction time, label) row with that target's own attributes.
# The task tables hold (target id, prediction time, label); take the train and validation splits.
tables = {split: task.get_table(split).df for split in ("train", "val")}
# Stack the two splits. pd.concat on a dict puts each split's name into the index;
# reset_index(level=0) turns that into an ordinary "split" column.
both = pd.concat(tables, names=["split"]).reset_index(level=0)
# Left join: every task row keeps its place in order and gains its target's columns from the entity table.
both = both.merge(entity.df, left_on=task.entity_col, right_on=entity.pkey_col, how="left")

# Features = everything except ids, the label, the prediction time, and the split marker.
drop = {"split", task.entity_col, entity.pkey_col, task.target_col, task.time_col}
X = both.drop(columns=[c for c in both.columns if c in drop])
# LightGBM needs numbers or categories, so convert the two column kinds it cannot use directly.
for col in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col] = X[col].astype("int64") // 10**9       # dates -> seconds since 1970
    elif X[col].dtype == "object":
        X[col] = X[col].astype("category")             # strings -> categories LightGBM can split on
# The 0/1 labels, and a True/False mask marking training rows (the rest are validation rows).
y = both[task.target_col].astype(int)
is_train = (both["split"] == "train").to_numpy()

# 300 small trees, each one correcting the errors of the trees before it. learning_rate shrinks each
# tree's contribution, num_leaves caps how big a tree can grow, verbose=-1 silences LightGBM's log.
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
# Fit on the training rows only, then score the validation rows.
clf.fit(X[is_train], y[is_train])
val_scores = clf.predict_proba(X[~is_train])[:, 1]     # probability of class 1 for each val row

# Report. task.evaluate needs the scores in the validation table's order, which the left join kept.
print("features used:", list(X.columns))
print("RelBench val metrics:", task.evaluate(val_scores, task.get_table("val")))
print(f"call lists of {args.list_size}:", call_list_metrics(y[~is_train], val_scores, args.list_size))