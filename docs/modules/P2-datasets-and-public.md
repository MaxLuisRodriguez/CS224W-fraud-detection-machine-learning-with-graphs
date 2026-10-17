# P2 · Which datasets, and are they public? (~30 min)

> **Handout bullets (§1.3):** "Which dataset(s) are you planning to use?" and "Is the dataset public? Do you plan to make it public before submitting the project? **[Note: this is a strict requirement]**"
>
> **Proposal section:** §2, first two sentences, plus the citations.

## 2.1 The answer

| Dataset · task | Role | Source |
|---|---|---|
| `rel-event` · `user-ignore` | Primary: **dense sharing** (outreach response) | [Hugging Face](https://huggingface.co/datasets/stanford-star/relbench-v1/tree/main/rel-event); originally the Kaggle [Event Recommendation Engine Challenge](https://www.kaggle.com/c/event-recommendation-engine-challenge) |
| `rel-avito` · `user-clicks` | Primary: **sparse sharing, large headroom** (intent signals) | [Hugging Face](https://huggingface.co/datasets/stanford-star/relbench-v1/tree/main/rel-avito); originally the Kaggle [Avito Context Ad Clicks](https://www.kaggle.com/c/avito-context-ad-clicks) |
| `rel-f1` · `driver-top3` | Debugging and preliminary model results (small enough for a laptop) | [Hugging Face](https://huggingface.co/datasets/stanford-star/relbench-v1/tree/main/rel-f1) |

**Public? Yes.** All three come from RelBench ([Robinson et al., NeurIPS 2024 D&B](https://arxiv.org/abs/2407.20060)). They're hosted openly on Hugging Face in `stanford-star/relbench-v1`: not gated, no login, and downloaded by one library call. The project never re-hosts data; the Colab downloads it. The Hugging Face card's license field reads "other", so cite RelBench and the original Kaggle sources.

## 2.2 Do: prove it downloads with no account

`scripts/peek_data.py`:

```python
"""P2: load a RelBench database and task with no login, and print what's in it."""
import argparse

from relbench import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
args = parser.parse_args()

dataset = load_dataset(args.dataset)   # downloads from Hugging Face the first time, then reads the local cache
db = dataset.get_db()                  # every table, with rows after the test cutoff removed
print(f"{args.dataset}: val cutoff {dataset.val_timestamp.date()}, test cutoff {dataset.test_timestamp.date()}")

# Schema: size, primary key, time column, and foreign keys (each foreign key becomes a graph edge type).
for name, table in db.table_dict.items():
    print(f"  {name:20s} {len(table.df):>10,} rows  pkey={table.pkey_col}  time={table.time_col}  "
          f"fkeys={table.fkey_col_to_pkey_table}")

# Task: which entity is scored, what label, how far ahead, and class balance.
task = dataset.load_task(args.task)
print(f"\n{args.task}: predict {task.target_col} for {task.entity_table}.{task.entity_col}, {task.timedelta} ahead")
for split in ("train", "val"):         # never look at test during development
    df = task.get_table(split).df
    print(f"  {split:5s} {len(df):>7,} rows | {df[task.time_col].nunique()} prediction dates | "
          f"positive rate {df[task.target_col].mean():.3f}")
```

```powershell
python scripts\peek_data.py --dataset rel-f1 --task driver-top3
```

**PASS** (tested Oct 16): nine tables, then:

```text
driver-top3: predict qualifying for drivers.driverId, 30 days 00:00:00 ahead
  train   1,353 rows | 59 prediction dates | positive rate 0.171
  val       588 rows | 27 prediction dates | positive rate 0.202
```

Don't run `peek_data.py` on `rel-event` or `rel-avito` on the laptop: `get_db()` loads every table into memory (30M+ rows), which is too much for 16 GB. P3 reads those two without loading them.

## 2.3 Do: start `proposal/refs.bib`

```bibtex
@inproceedings{robinson2024relbench,
  title     = {{RelBench}: A Benchmark for Deep Learning on Relational Databases},
  author    = {Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan E. and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Datasets and Benchmarks Track},
  year      = {2024}
}
```

This is the citation the Hugging Face README asks for. Add the others from each paper's arXiv "Export BibTeX" link as you go: Fey et al. 2024 (RDL), Hamilton et al. 2017 (GraphSAGE), Fey & Lenssen 2019 (PyG), Zhou 2026, CAMA, Bilgic & Getoor 2010. The links are in [RELATED_WORK.md](../RELATED_WORK.md).

## 2.4 Do: write proposal §2, opening

Two sentences in `proposal/notes.md`: which datasets and tasks, and the public-availability statement above.

**Self-check:** why is `rel-f1` in the proposal at all, given it isn't a sales dataset?

**Next: [P3 · Describe the dataset, task, and metric](P3-dataset-task-metric.md)**
