# P7 · Why the model fits the data, and preliminary results (~2 h, mostly training in the background)

> **Handout bullet (§1.3):** "Why is/are the model(s) appropriate for the dataset(s) you have chosen?"
>
> **Proposal sections this feeds:** §3, final paragraph, and §4, preliminary results.

The strongest answer to "why is this model appropriate" is evidence, so this module produces it. You'll measure the zero-lookup reference (LightGBM on each target's own row), prove the sampler can't see the future, then train the GNN with and without partial-observation training and average over seeds. Everything runs on `rel-f1` on the laptop.

**Keep the laptop plugged in with the lid open while training.** Closing the lid puts Windows into Modern Standby, which pauses the run: a test run on Oct 17 stalled for 90 minutes that way, and the `minutes` field in the JSON then includes the sleep.

## 7.1 Zero lookups: `scripts/baseline_lightgbm.py`

LightGBM sees only each target's own row (the `drivers` table), joined onto the task table. It's the left end of every budget frontier: what you can do with no lookups at all.

```python
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
entity = db.table_dict[task.entity_table]  # the prospects' own table: one row of attributes per prospect

# Features = the target's own row, joined onto each (target, prediction time, label) row of the task table.
tables = {split: task.get_table(split).df for split in ("train", "val")}
both = pd.concat(tables, names=["split"]).reset_index(level=0)
both = both.merge(entity.df, left_on=task.entity_col, right_on=entity.pkey_col, how="left")
drop = {"split", task.entity_col, entity.pkey_col, task.target_col, task.time_col}
X = both.drop(columns=[c for c in both.columns if c in drop])
for col in X.columns:
    if pd.api.types.is_datetime64_any_dtype(X[col]):
        X[col] = X[col].astype("int64") // 10**9           # timestamps -> seconds since 1970
    elif X[col].dtype == "object":
        X[col] = X[col].astype("category")               # strings -> categories LightGBM can split on
y = both[task.target_col].astype(int)
is_train = (both["split"] == "train").to_numpy()

clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
clf.fit(X[is_train], y[is_train])
val_scores = clf.predict_proba(X[~is_train])[:, 1]

print("features used:", list(X.columns))
print("RelBench val metrics:", task.evaluate(val_scores, task.get_table("val")))
print(f"call lists of {args.list_size}:", call_list_metrics(y[~is_train], val_scores, args.list_size))
```

```powershell
python scripts\baseline_lightgbm.py
```

**PASS** (tested Oct 17; a few seconds):

```text
features used: ['driverRef', 'code', 'forename', 'surname', 'dob', 'nationality']
RelBench val metrics: {'roc_auc': 0.6606493343606099}
call lists of 128: {'auroc': 0.6613042156279608, 'ap': 0.3474319114668765, 'p@13': 0.23576923076923076, 'lists_scored': 200}
```

LightGBM is deterministic here, so your numbers should match. A warning about unauthenticated Hugging Face requests is harmless.

**Say this in the proposal:** four of the six features (`driverRef`, `code`, `forename`, `surname`) identify the driver. LightGBM can learn "this driver often qualifies top-3" from the training dates and reuse it on the validation dates. That's identity memorization, not "attributes". The GNN's driver encoder sees the same columns, so the comparison is fair, but it's worth one sentence.

## 7.2 The sampler can't see the future: `tests/test_no_leakage.py`

```python
"""P7: every sampled row is dated no later than the prediction time of the target it was sampled for."""
import pytest

from wtl.data import build_graph, column_stypes, load, make_loader


@pytest.fixture(scope="module")
def batch_and_task():
    dataset, task, db = load("rel-f1", "driver-top3")
    data, _ = build_graph(db, "rel-f1", task, column_stypes(db, "rel-f1"))
    loader, _ = make_loader(data, task, "val", batch_size=64)
    return next(iter(loader)), task


def test_no_rows_from_the_future(batch_and_task):
    batch, task = batch_and_task
    seed_time = batch[task.entity_table].seed_time        # prediction time of each target
    checked = 0
    for node_type in batch.node_types:
        store = batch[node_type]
        if "time" not in store or store.num_nodes == 0:
            continue                                       # tables without timestamps can't leak the future
        owner = store.batch                                # which target's neighborhood each row belongs to
        assert (store.time <= seed_time[owner]).all(), f"{node_type} contains rows after the prediction time"
        checked += 1
    assert checked > 0, "no timestamped tables were checked"
```

`store.batch` exists because sampling is disjoint (P5): every sampled row belongs to exactly one target's neighborhood, so it can be checked against that target's own prediction time.

```powershell
python -m pytest tests
```

**PASS** (~11 s):

```text
tests\test_metrics.py ..                                                 [ 66%]
tests\test_no_leakage.py .                                               [100%]

============================= 3 passed in 10.79s ==============================
```

**STOP:** an `AssertionError` naming a table means the loader is ignoring time. Check `time_attr="time"` and `input_time=table_input.time` in `data.py`.

What the test does **not** check: static tables (no time column) can't be tested this way, and it looks at one batch of 64 validation targets, not every batch. Columns that would leak the label are RelBench's job (`remove_columns=task.hidden_columns()` in `data.py`).

## 7.3 `scripts/train_rdl.py`

One script trains the GNN and then evaluates it while hiding a growing share of the graph.

```python
"""P7: train the relational GNN, then evaluate it while hiding a growing share of the graph.

--train_reveal full    trains on complete neighborhoods (the standard RelBench recipe)
--train_reveal random  hides a random share of rows in every training batch (partial-observation training)
--hide drop|mask       hidden rows are removed from message passing, or kept as [MASK] stubs
Evaluation reveals each distinct row with probability p, for every p in --eval_p (1.0 = full graph).
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from relbench.base import TaskType
from torch.nn import BCEWithLogitsLoss
from torch_geometric.seed import seed_everything

from wtl.data import build_graph, column_stypes, load, make_loader
from wtl.metrics import call_list_metrics
from wtl.model import RDLModel
from wtl.reveal import random_reveal


@torch.no_grad()
def predict(model, loader, entity_table: str, p: float, hide: str, device) -> "numpy.ndarray":
    """Probabilities for every row of the loader's table, in table order, seeing a share p of the rows."""
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        revealed = None if p >= 1.0 else random_reveal(batch, entity_table, p)
        probs.append(torch.sigmoid(model(batch, entity_table, revealed=revealed, hide=hide)).cpu())
    return torch.cat(probs).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    parser.add_argument("--train_reveal", choices=["full", "random"], default="full")
    parser.add_argument("--hide", choices=["drop", "mask"], default="drop")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_p", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    args = parser.parse_args()

    seed_everything(args.seed)                                    # Python, NumPy, and torch RNGs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, task, db = load(args.dataset, args.task)
    assert task.task_type == TaskType.BINARY_CLASSIFICATION, "this script handles binary tasks"
    data, col_stats = build_graph(db, args.dataset, task, column_stypes(db, args.dataset))
    train_loader, _ = make_loader(data, task, "train", batch_size=args.batch_size, shuffle=True)
    val_loader, val_table = make_loader(data, task, "val", batch_size=args.batch_size)

    model = RDLModel(data, col_stats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = BCEWithLogitsLoss()                                 # binary cross-entropy on raw logits
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            revealed = None
            if args.train_reveal == "random":
                revealed = random_reveal(batch, task.entity_table, p=random.random())  # a new share each batch
            logits = model(batch, task.entity_table, revealed=revealed, hide=args.hide)
            loss = loss_fn(logits, batch[task.entity_table].y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * logits.numel()
            count += logits.numel()
        print(f"epoch {epoch:02d}  train loss {loss_sum / count:.4f}")

    # Evaluate the final model at every revelation level (fixed seed for the reveal draw; the sampler is unseeded).
    results = {"args": vars(args), "minutes": None, "val": {}}
    y_val = val_table.df[task.target_col].astype(int).to_numpy()
    for p in args.eval_p:
        torch.manual_seed(1234)
        scores = predict(model, val_loader, task.entity_table, p, args.hide, device)
        official = task.evaluate(scores, val_table)               # RelBench's own metrics (roc_auc, ...)
        lists = call_list_metrics(y_val, scores, list_size=min(128, len(y_val)))
        results["val"][str(p)] = {"roc_auc": official["roc_auc"], **{f"list_{k}": v for k, v in lists.items()}}
        print(f"reveal p={p:.2f}  val AUROC {official['roc_auc']:.4f}  call-list {lists}")
    results["minutes"] = (time.time() - start) / 60

    out = Path("outputs") / args.dataset / args.task / f"rdl_{args.train_reveal}_{args.hide}_ep{args.epochs}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
```

| Flag | Values | Meaning |
|---|---|---|
| `--train_reveal` | `full` | Standard training on complete neighborhoods: the **mask-naive** model of H2 |
| | `random` | Each training batch draws ρ ~ U(0, 1) and reveals each distinct row with probability ρ: Eq. (6) from P6 |
| `--hide` | `drop` / `mask` | How hidden rows are treated, in training and evaluation (P6 §6.1) |
| `--eval_p` | default `1 0.75 0.5 0.25 0` | Revelation levels for evaluation; each row is revealed at random with probability *p* |

Every run writes `outputs/rel-f1/driver-top3/rdl_<train_reveal>_<hide>_ep<epochs>_seed<seed>.json` (Git-ignored).

First run, the standard model with 20 epochs (7.4 explains why 20):

```powershell
python scripts\train_rdl.py --train_reveal full --hide drop --epochs 20 --seed 0
```

**PASS** (tested Oct 17; ~6.5 min). After two harmless warnings (a torch note about redirects on Windows, and the Hugging Face one):

```text
epoch 01  train loss 0.6110
epoch 02  train loss 0.4467
epoch 03  train loss 0.4497
epoch 04  train loss 0.4424
epoch 05  train loss 0.4426
epoch 06  train loss 0.4362
epoch 07  train loss 0.4322
epoch 08  train loss 0.4210
epoch 09  train loss 0.3995
epoch 10  train loss 0.3803
epoch 11  train loss 0.3601
epoch 12  train loss 0.3332
epoch 13  train loss 0.3231
epoch 14  train loss 0.3114
epoch 15  train loss 0.3164
epoch 16  train loss 0.2926
epoch 17  train loss 0.2841
epoch 18  train loss 0.2787
epoch 19  train loss 0.2709
epoch 20  train loss 0.2686
reveal p=1.00  val AUROC 0.7736  call-list {'auroc': 0.7737375739329658, 'ap': 0.49652451638186806, 'p@13': 0.578076923076923, 'lists_scored': 200}
reveal p=0.75  val AUROC 0.7771  call-list {'auroc': 0.7769577950271253, 'ap': 0.501798687816351, 'p@13': 0.5892307692307692, 'lists_scored': 200}
reveal p=0.50  val AUROC 0.7764  call-list {'auroc': 0.7754293329395443, 'ap': 0.5050144995550945, 'p@13': 0.5846153846153846, 'lists_scored': 200}
reveal p=0.25  val AUROC 0.7609  call-list {'auroc': 0.7600031889603991, 'ap': 0.49330960558347214, 'p@13': 0.545, 'lists_scored': 200}
reveal p=0.00  val AUROC 0.6562  call-list {'auroc': 0.6583241763365015, 'ap': 0.29291912961052324, 'p@13': 0.18384615384615388, 'lists_scored': 200}
saved outputs\rel-f1\driver-top3\rdl_full_drop_ep20_seed0.json
```

Your numbers **will differ**, even with `--seed 0`: the neighbor sampler doesn't follow the seed, so no two runs see the same neighborhoods. What should hold: the loss sits near 0.44 for several epochs, then falls to about 0.27–0.31 by epoch 20, and p = 1 AUROC lands somewhere around 0.73–0.78.

**STOP:**

- The loss is still above 0.40 at epoch 20: that run never left the plateau (7.4). Note it, and run another seed.
- `TypeError` about `revealed`: `model.py` is still P5's version; use P6's.

## 7.4 One run is not a result

Two things showed up in testing, and both decide how you report numbers.

**Training sits on a plateau first.** For the first several epochs the loss stays near 0.44, close to 0.457, the loss of always predicting the 17% base rate. Then the model starts to learn. With 10 epochs, that's a knife-edge. Three test runs of `--train_reveal full --hide drop --epochs 10`:

| Seed | Train loss, epoch 10 | Val AUROC, p = 1 | Val AUROC, p = 0 |
|---|---:|---:|---:|
| 0 | 0.357 | 0.760 | 0.596 |
| 1 | 0.354 | 0.775 | 0.637 |
| 2 | **0.434** | **0.617** | 0.465 |

Seed 2 never left the plateau, and it scored *below* LightGBM with every row visible. At 20 epochs, all six test runs (two configurations × three seeds) left the plateau, though two of them only got off it around epochs 11–13.

**The same seed doesn't reproduce.** `--train_reveal full --hide drop --epochs 10 --seed 0`, run twice on Oct 17 with identical code, scored 0.760 and 0.745 at p = 1, and 0.771 and 0.733 at p = 0.5. The neighbor sampler doesn't follow the seed, so each run trains on different neighborhoods.

So: **20 epochs, three seeds, mean ± standard deviation, and the same number of epochs for every configuration you compare.** The single-seed table you may have seen earlier (with "full-trained collapses to 0.489 at p = 0") broke two of those rules: it compared a 10-epoch full-trained run against a 20-epoch random-trained run, one seed each.

## 7.5 The seed sweep

Five more runs, ~6.5 min each (about 35 min). Start it, then write 7.8 while it runs.

```powershell
python scripts\train_rdl.py --train_reveal random --hide drop --epochs 20 --seed 0
foreach ($seed in 1, 2) {
    python scripts\train_rdl.py --train_reveal full --hide drop --epochs 20 --seed $seed
    python scripts\train_rdl.py --train_reveal random --hide drop --epochs 20 --seed $seed
}
```

Optional, if you have another 40 minutes: the same six runs with `--hide mask`. Mask mode was only tested at 10 epochs, one seed each (p = 1 AUROC 0.739 full-trained, 0.734 random-trained), so there are no reference numbers for it at 20 epochs.

## 7.6 `scripts/summarize_runs.py`: average the seeds

```python
"""P7: average train_rdl.py runs over seeds and print a markdown table for the proposal.

Reads every outputs/<dataset>/<task>/rdl_*_ep*_seed*.json and groups runs by training configuration.
Each cell is the validation AUROC at one revelation level p: mean +/- standard deviation over seeds.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
parser.add_argument("--metric", default="roc_auc")   # or list_auroc, list_ap, list_p@13 (call-list metrics)
args = parser.parse_args()

rows = []
for path in sorted((Path("outputs") / args.dataset / args.task).glob("rdl_*_ep*_seed*.json")):
    run = json.loads(path.read_text())
    a = run["args"]
    config = f"{a['train_reveal']}, {a['hide']}, {a['epochs']} ep"   # one row of the table per configuration
    for p, metrics in run["val"].items():
        rows.append({"config": config, "seed": a["seed"], "p": float(p), "value": metrics[args.metric]})
df = pd.DataFrame(rows)

stats = df.groupby(["config", "p"]).value.agg(["mean", "std"])
seeds = df.groupby("config").seed.unique()
ps = sorted(df.p.unique(), reverse=True)   # columns from full information (p = 1.0) down to nothing (p = 0.0)

print(f"{args.dataset} {args.task}: val {args.metric}, mean +/- sd over seeds\n")
print("| Config | Seeds | " + " | ".join(f"p = {p:g}" for p in ps) + " |")
print("|---|---|" + "---:|" * len(ps))
for config in stats.index.get_level_values("config").unique():
    cells = []
    for p in ps:
        mean, sd = stats.loc[(config, p)]
        cells.append(f"{mean:.3f}" if pd.isna(sd) else f"{mean:.3f} +/- {sd:.3f}")   # sd is NaN for one seed
    print(f"| {config} | {','.join(map(str, sorted(seeds[config])))} | " + " | ".join(cells) + " |")
```

```powershell
python scripts\summarize_runs.py
python scripts\summarize_runs.py --metric "list_p@13"
```

**PASS** (tested Oct 17, three seeds each). The output is a Markdown table you can paste:

```text
rel-f1 driver-top3: val roc_auc, mean +/- sd over seeds

| Config | Seeds | p = 1 | p = 0.75 | p = 0.5 | p = 0.25 | p = 0 |
|---|---|---:|---:|---:|---:|---:|
| full, drop, 20 ep | 0,1,2 | 0.752 +/- 0.020 | 0.758 +/- 0.018 | 0.746 +/- 0.027 | 0.749 +/- 0.010 | 0.642 +/- 0.049 |
| random, drop, 20 ep | 0,1,2 | 0.734 +/- 0.015 | 0.735 +/- 0.019 | 0.740 +/- 0.026 | 0.718 +/- 0.019 | 0.612 +/- 0.025 |
```

```text
rel-f1 driver-top3: val list_p@13, mean +/- sd over seeds

| Config | Seeds | p = 1 | p = 0.75 | p = 0.5 | p = 0.25 | p = 0 |
|---|---|---:|---:|---:|---:|---:|
| full, drop, 20 ep | 0,1,2 | 0.463 +/- 0.100 | 0.478 +/- 0.097 | 0.457 +/- 0.111 | 0.473 +/- 0.067 | 0.147 +/- 0.075 |
| random, drop, 20 ep | 0,1,2 | 0.389 +/- 0.052 | 0.388 +/- 0.069 | 0.408 +/- 0.134 | 0.405 +/- 0.180 | 0.185 +/- 0.112 |
```

Your means will move by a few hundredths. **Use your own table in the proposal**, not this one.

**STOP:** a row with fewer than three seeds means a run crashed or was interrupted. Check which JSON is missing in `outputs\rel-f1\driver-top3\` and rerun that seed.

## 7.7 Read the results

These are the conclusions from the test runs. Check that yours point the same way before you write them down.

**The graph helps on `rel-f1`.** With every row visible, the GNN reaches 0.752 ± 0.020 val AUROC against LightGBM's 0.661 on the target's own row, and Precision@13 of 0.463 against 0.236 (base rate 0.202).

**H1, a concave frontier: supported, under random revelation.** For the full-trained model, revealing a random 25% of rows recovers (0.749 − 0.642) / (0.752 − 0.642) ≈ **97%** of the zero-to-full AUROC gap. For the random-trained model it's ≈ 87%. Caveats: rows were revealed at random, not chosen; and `rel-f1` neighborhoods are redundant (a driver has dozens of results and standings rows), so 25% of them still says a lot.

**H2, partial-observation training closes a low-budget gap: not supported on `rel-f1`.** With both models trained for 20 epochs, the full-trained model degrades gracefully, and random-revelation training is no better at any *p*. At p = 0.25 it's behind (0.718 ± 0.019 vs. 0.749 ± 0.010). The earlier impression that full-trained models collapse came from 10-epoch runs barely past the plateau. What this means:

- It's a real finding, and it fits known results: KumoRFM-2 reports robustness to **random** feature and foreign-key removal on relational data. That's why the project's claim is about **chosen, budgeted** lookups (C1 in [RELATED_WORK.md](../RELATED_WORK.md)), where a policy concentrates its budget on some prospects and tables. Random revelation never does that.
- H2 stays open for `mask` mode, for policy-chosen lookups (B3), and on the primary datasets (B1).

**At p = 0 both GNNs rank the top of the list badly.** Precision@13 falls to 0.15–0.19, *below* the 0.202 base rate, even though AUROC is 0.61–0.64. With no relational context, LightGBM (P@13 0.236) is the better zero-lookup model, and that's the reference the frontiers should start from.

## 7.8 Why the model fits these datasets

The argument for proposal §3, each point with its evidence:

1. **The data is relational, and the model uses the relations directly.** Message passing runs over the foreign keys, so a prospect's score depends on the rows linked to it. Evidence: 0.752 vs. 0.661 on `rel-f1` (7.6); GNN–LightGBM gaps of +0.031 and +0.118 on the primary tasks (leaderboard, P4).
2. **Temporal sampling rules out leakage by construction.** Each target sees only rows dated at or before its own prediction time. Evidence: the leakage test (7.2).
3. **Shared entities are literally shared rows.** The same race or event sampled for several targets carries the same `n_id`, so one fetch can reveal it to all of them. Evidence: 631 sampled race rows were 178 distinct races (P5); `check_reveal.py` shows zero inconsistent rows (P6).
4. **Partial observation is native to the model.** Hiding a row means dropping its node's edges, or swapping its vector for `[MASK]`; the same trained model runs at every revelation level unchanged. Evidence: the tables in 7.6.

And one honest limit, in one sentence: so far, rows are hidden at random on one small dataset (588 validation rows), so differences under about 0.03 AUROC are within seed noise.

## 7.9 Fig. 2 (optional)

Run the script you wrote in P6 §6.7 on your two three-seed configurations:

```powershell
python scripts\fig_reveal_curve.py --configs full_drop_ep20 random_drop_ep20
```

**PASS:** `saved proposal\figures\fig2_reveal_curve.pdf`. Two lines with error bars, blue (full-trained) above orange (random-trained) from 25% to 100%, both dropping at 0%, and the gray LightGBM line at 0.661 between the two p = 0 points and the rest.

Caption draft (update the numbers to yours):

> *Figure 2: Validation AUROC on `rel-f1` `driver-top3` (588 rows) when each row of a target's sampled neighborhood is revealed at random with probability p. Mean ± s.d. over 3 seeds; 20 training epochs each. Full-trained: standard training on complete neighborhoods. Random-trained: each training batch reveals rows with a random rate. Gray: LightGBM on the target's own row.*

## 7.10 Do: write proposal §3 (final paragraph) and §4

In `proposal/notes.md`:

1. **§3, final paragraph:** the four points from 7.8 in 4–5 sentences, plus the one-sentence limit.
2. **§4, evaluation plan:** budget frontiers for each policy (random, case-local round-robin, uncertainty-first, coverage, a Zhou-style singleton scorer, the learned batch value, oracle-greedy, full information; see [PROJECT_PLAN §5](../PROJECT_PLAN.md#5-method)), the analyses list ([§6](../PROJECT_PLAN.md#6-experiments-and-analyses)), and H1–H5 in one line each. Final results will use five seeds and bootstrap confidence intervals over call lists.
3. **§4, preliminary results:** the sharing statistics (P4); LightGBM vs. GNN on `rel-f1` (7.1, 7.6); and the H1/H2 reading from 7.7, labeled as *`rel-f1` validation, 3 seeds, random revelation*. A small table (or Fig. 2) with your own numbers.

**Self-check**

1. Why can't a single run tell you whether random-trained beats full-trained at p = 0.25?
2. LightGBM's features include driver names. Why does that matter when you compare it with the GNN, and how would you test it?
3. What does the leakage test check, and what doesn't it check?

**Next: [P8 · Assemble and submit the proposal](P8-assemble-and-submit.md)**
