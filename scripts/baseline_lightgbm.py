"""P7: LightGBM on each target's own row only. This is the "zero lookups" reference point."""
import argparse

import lightgbm as lgb
import pandas as pd

from relbench import load_dataset

from wtl.metrics import call_list_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--list_size", type=int, default=128)
args = parser.parse_args()

dataset = load_dataset(args.dataset)
task = dataset.load_task(args.task)
db = dataset.get_db()
entity = db.table_dict[task.entity_table]   # the targets' own table (drivers)

# Join each (target, prediction time, label) row with that target's own attributes.
tables = {split: task.get_table(split).df for split in ("train", "val")}
both = pd.concat(tables, names=["split"]).reset_index(level=0)
both = both.merge(entity.df, left_on=task.entity_col, right_on=entity.pkey_col, how="left")

# Features = everything except ids, the label, the prediction time, and the split marker.
drop = {"split", task.entity_col, entity.pkey_col, task.target_col, task.time_col}
X = both.drop(columns=[c for c in both.columns if c in drop])
for col in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col] = X[col].astype("int64") // 10**9       # dates -> seconds since 1970
    elif X[col].dtype == "object":
        X[col] = X[col].astype("category")             # strings -> categories LightGBM can split on
y = both[task.target_col].astype(int)
is_train = (both["split"] == "train").to_numpy()

clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
clf.fit(X[is_train], y[is_train])
val_scores = clf.predict_proba(X[~is_train])[:, 1]     # probability of class 1 for each val row

print("features used:", list(X.columns))
print("RelBench val metrics:", task.evaluate(val_scores, task.get_table("val")))
print(f"call lists of {args.list_size}:", call_list_metrics(y[~is_train], val_scores, args.list_size))