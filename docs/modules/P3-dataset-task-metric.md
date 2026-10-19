# P3 · Describe the dataset, task, and metric (~1 h)

> **Handout bullet (§1.3):** "Describe the dataset/task/metric. Cite as relevant, especially the data."
>
> **Proposal section this feeds:** §2, Dataset, task, metric (~½ page, shared with P4).

You'll print a dataset card for each of the three tasks without downloading the large databases, read each schema and label definition, and implement the call-list metrics with a test. Everything below was tested on this laptop on Oct 17.

## 3.1 `scripts/dataset_card.py`

RelBench stores a `manifest.yaml` per database and per task on Hugging Face: keys, time columns, split cutoffs, and the label's SQL. A parquet file's row count sits in its footer, so the script reads a few KB per table instead of downloading 30M-row files.

```python
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
```

The output is Markdown, so save each card where you can paste from it. `Out-File -Encoding utf8` matters: PowerShell 5's plain `>` writes UTF-16, which Git treats as binary.

```powershell
New-Item -ItemType Directory -Force proposal\cards | Out-Null
python scripts\dataset_card.py --dataset rel-event --task user-ignore | Out-File -Encoding utf8 proposal\cards\rel-event.md
python scripts\dataset_card.py --dataset rel-avito --task user-clicks | Out-File -Encoding utf8 proposal\cards\rel-avito.md
python scripts\dataset_card.py --dataset rel-f1 --task driver-top3 | Out-File -Encoding utf8 proposal\cards\rel-f1.md
Get-Content proposal\cards\rel-event.md
```

**PASS** (tested Oct 17; ~10 s per card): `rel-event.md` reads

```text
## rel-event: Event recommendation: users, events, attendance records, and social and interest signals.

Validation cutoff 2012-11-21, test cutoff 2012-11-29.

| Table | Rows | Primary key | Time column | Foreign keys |
|---|---:|---|---|---|
| `user_friends` | 30,386,403 | none | none | user -> users, friend -> users |
| `event_interest` | 15,398 | none | timestamp | event -> events, user -> users |
| `event_attendees` | 11,245,010 | none | start_time | event -> events, user_id -> users |
| `users` | 38,209 | user_id | joinedAt | none |
| `events` | 3,137,972 | event_id | start_time | user_id -> users |

### Task `user-ignore`: Predict whether a user will ignore more than 2 event invitations in the next 7 days.

Type binary_classification; entity `users.user`; label `target`; horizon 7 days.

| Split | Rows | Prediction dates | Positive rate |
|---|---:|---:|---:|
| train | 19,239 | 22 | 0.169 |
| val | 2,013 | 1 | 0.111 |
| test | 1,958 | held out | held out |
```

The task tables in the other two cards:

```text
rel-avito  user-clicks: entity UserInfo.UserID; label num_click; horizon 4 days
| train | 59,454 | 3 | 0.039 |
| val | 21,183 | 1 | 0.035 |
| test | 47,996 | held out | held out |

rel-f1  driver-top3: entity drivers.driverId; label qualifying; horizon 30 days
| train | 1,353 | 59 | 0.171 |
| val | 588 | 27 | 0.202 |
| test | 726 | held out | held out |
```

**STOP:**

- `HfHubHTTPError` or a timeout: check your connection and rerun. No login is needed; the HF warning about unauthenticated requests is harmless.
- The card file opens as gibberish in VS Code: it was written without `-Encoding utf8`. Rerun that line.

## 3.2 Read the three schemas

Open the cards and check each statement below against them. Each one becomes a sentence in your proposal.

**`rel-event`** (entity `users`, 38,209 rows)

- Four tables (five foreign keys) point at `users`: `event_attendees.user_id` (11.2M attendance records), `user_friends.user` and `.friend` (30.4M friendship rows), `event_interest.user` (15k interest clicks), and `events.user_id` (the event's creator).
- Two hops from a user: user → attendance row → **event**, and user → friendship row → **friend**. These are the shared entities you'll measure in P4.
- `user_friends` has **no time column**, so RelBench treats every friendship as always known. Say so in the proposal; it's a limitation of the data.

**`rel-avito`** (entity `UserInfo`, 98,250 rows, no time column)

- Three tables point at users: `SearchInfo` (2.6M searches), `VisitStream` (6.5M ad visits), and `PhoneRequestsStream` (303k "show me the seller's number" requests).
- `SearchStream` (9.3M ads shown in search results, with an `IsClick` flag) points at `SearchInfo` and `AdsInfo`, not at users. A user's **past clicks are two hops away**: user → search → shown ad. That matters for H3.
- `AdsInfo`, `Category`, `Location`, and `UserInfo` are static (no time column).

**`rel-f1`** (entity `drivers`, 857 rows)

- `results`, `standings`, and `qualifying` point at drivers; all three are dated.
- `drivers`, `constructors`, and `circuits` are static.

## 3.3 Labels and temporal splits

Each label is defined by SQL in the task's `manifest.yaml`. Read it on Hugging Face, e.g. [rel-event/tasks/user-ignore/manifest.yaml](https://huggingface.co/datasets/stanford-star/relbench-v1/blob/main/rel-event/tasks/user-ignore/manifest.yaml).

| Task | Label = 1 when, in the horizon after the prediction date … | Who gets a row |
|---|---|---|
| `rel-event` · `user-ignore` (7 days) | the user has **more than 2** attendance records with status `invited` (invited, never answered) | users with any attendance record for an event in the window |
| `rel-avito` · `user-clicks` (4 days) | the user's searches produce **more than one** click (`SearchStream.IsClick = 1`) | users who searched in the window |
| `rel-f1` · `driver-top3` (30 days) | the driver's best qualifying position is **1–3** | drivers who qualify in the window and also qualified in the past year |

In `user-ignore` the positive class is the *unresponsive* prospect, so the top of that ranking is who to skip or contact another way. The metric is the same either way.

**Splits.** The splits are cut by time. Train prediction dates step back from the validation cutoff one horizon at a time, so the last training window ends exactly at the cutoff. Validation dates start at the validation cutoff, and their windows end by the test cutoff. Test starts at the test cutoff. Prediction dates per split (from the cards): 22 / 1 train / val dates on `rel-event`, 3 / 1 on `rel-avito`, and 59 / 27 on `rel-f1`. So the validation sets of the two primary datasets are a **single day** each: one day's call list, which is exactly the setting of the project.

Development uses validation only. The card counts test rows but never reads test labels.

## 3.4 The metrics

| Metric | What it measures | Why it's here |
|---|---|---|
| **AUROC** | Probability that a random positive is scored above a random negative; 0.5 = random | RelBench's official metric for these tasks, so results compare with the leaderboard |
| **Average precision (AP)** | Area under the precision–recall curve; rewards putting positives at the very top | Positives are rare on `rel-avito` (3.5%), where AUROC can look fine while the top of the list is poor |
| **Precision@K over call lists** | Draw a random list of *N* = 128 validation prospects, rank them, take the top *K* = ⌈0.1·*N*⌉ = **13**, and count the fraction that are positive. Average over 200 lists | The rep's view: of the calls I make first, how many were the right ones? A random ranking scores the base rate (0.202 on `rel-f1` val) |
| **Budget frontier** | Each metric as a function of lookups per prospect (0, ½, 1, 2, 4, 8, all), one curve per lookup policy | The headline result of the project. You'll see a first version in P7, where rows are revealed at random instead of chosen |

## 3.5 `src/wtl/metrics.py` and its test

```python
"""Ranking metrics for call lists: the rep calls the top-K prospects first."""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(y_true, scores, k: int) -> float:
    """Fraction of real positives among the k highest-scored prospects."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    top = np.argsort(-scores, kind="stable")[:k]   # indices of the k highest scores
    return float(y_true[top].mean())


def call_list_metrics(y_true, scores, list_size: int = 128, num_lists: int = 200, seed: int = 0) -> dict:
    """Average AUROC, average precision, and Precision@K over random call lists.

    K = ceil(0.1 * list_size). AUROC and AP are undefined for a list with no positives (or no negatives),
    so those lists are skipped for those two metrics; "lists_scored" says how many counted.
    """
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    k = math.ceil(0.1 * list_size)
    auroc, ap, p_at_k = [], [], []
    for _ in range(num_lists):
        idx = rng.choice(len(y_true), size=list_size, replace=False)
        y, s = y_true[idx], scores[idx]
        p_at_k.append(precision_at_k(y, s, k))
        if 0 < y.sum() < len(y):
            auroc.append(roc_auc_score(y, s))
            ap.append(average_precision_score(y, s))
    return {"auroc": float(np.mean(auroc)), "ap": float(np.mean(ap)), f"p@{k}": float(np.mean(p_at_k)),
            "lists_scored": len(auroc)}
```

`tests/test_metrics.py`:

```python
"""P3: the call-list metric does what we claim."""
from wtl.metrics import precision_at_k


def test_perfect_ranking_puts_positives_on_top():
    y = [1, 0, 1, 0, 0]
    scores = [0.9, 0.1, 0.8, 0.3, 0.2]
    assert precision_at_k(y, scores, k=2) == 1.0


def test_worst_ranking_puts_positives_last():
    y = [1, 0, 1, 0, 0]
    scores = [0.1, 0.9, 0.2, 0.8, 0.7]
    assert precision_at_k(y, scores, k=2) == 0.0
```

```powershell
python -m pytest tests\test_metrics.py
# Sanity check on fake data: random scores should give AUROC ~0.5 and P@13 ~ the base rate (0.2).
python -c "import numpy as np; from wtl.metrics import call_list_metrics; rng = np.random.default_rng(1); y = rng.random(20000) < 0.2; print('random :', call_list_metrics(y, rng.random(20000))); print('perfect:', call_list_metrics(y, y + 0.0))"
```

**PASS:**

```text
tests\test_metrics.py ..                                                 [100%]
============================== 2 passed in 6.66s ==============================
random : {'auroc': 0.5029368574850099, 'ap': 0.23406582062701559, 'p@13': 0.21615384615384614, 'lists_scored': 200}
perfect: {'auroc': 1.0, 'ap': 1.0, 'p@13': 0.9996153846153846, 'lists_scored': 200}
```

Perfect scores give P@13 just below 1: a few random lists hold fewer than 13 positives, so even a perfect ranking can't fill the top 13. On `rel-avito` (3.5% positives, ~4.5 per list of 128) that ceiling is low, so always report P@K next to the base rate.

One more property to know: all 200 lists are drawn from the same validation rows, so they aren't independent samples. With only 588 rows on `rel-f1`, the list average tracks the whole-split number closely (you'll see this in P7).

**STOP:**

- `ModuleNotFoundError: No module named 'wtl'`: the editable install from P0 step 7 is missing. Run `uv pip install -e .`.
- `pytest` collects 0 tests: run from the repo root, and check the file is `tests\test_metrics.py`.

## 3.6 Do: write proposal §2, dataset, task, and metric

In `proposal/notes.md`, under your §2 opening from P2:

1. **One table** with a row per task: dataset, entity, label in words, horizon, train / val / test rows, val positive rate. Copy the numbers from `proposal/cards/`.
2. **One sentence per schema**: what the entity is, which tables point at it, and which entities sit two hops away (events and friends; ads, categories, locations).
3. **Splits**: temporal; validation is a single day on both primary datasets; test labels unused until the end.
4. **Metrics**: AUROC (official), AP, and Precision@13 over random lists of 128, plus the budget frontier as the headline. Define *K* = ⌈0.1·*N*⌉ in words: the calls a rep makes first.
5. **Citations**: RelBench (`robinson2024relbench`, already in `refs.bib`), and the original sources from P2's table (the two Kaggle competitions). Add BibTeX entries for those two:

```bibtex
@misc{kaggle2013event,
  title        = {Event Recommendation Engine Challenge},
  howpublished = {Kaggle competition, \url{https://www.kaggle.com/c/event-recommendation-engine-challenge}},
  year         = {2013}
}

@misc{kaggle2015avito,
  title        = {Avito Context Ad Clicks},
  howpublished = {Kaggle competition, \url{https://www.kaggle.com/c/avito-context-ad-clicks}},
  year         = {2015}
}
```

Open each Kaggle page once to confirm the title and year before you submit.

**Self-check**

1. On `rel-avito`, how many hops is a user's past click from the user, and why does that matter for H3?
2. Why compute Precision@K over random lists of 128 instead of over the whole validation split?
3. Why does a perfect ranking score P@13 slightly below 1, and why is that worse on `rel-avito`?

**Next: [P4 · Why these datasets?](P4-why-these-datasets.md)**
