# Worth the Lookup

**Learning what to research before the call: budgeted relational retrieval for call-list prioritization.** Stanford CS224W project (Fall 2026), **project type 1: Real-world application of GNNs** (blog post + Colab).

A sales rep's call list is a graph. Prospects share accounts, events, colleagues, and activity, so one lookup can sharpen several predictions. Lookups cost money (enrichment credits), tokens (LLM research agents), and time. This project asks:

> Given a call list of *N* prospects and a budget of *B* lookups, which lookups should a research agent make so that a graph neural network ranks the list best?

- **Data:** public [RelBench](https://relbench.stanford.edu/) relational databases (auto-downloaded; nothing collected by hand).
- **Models:** PyTorch Geometric heterogeneous temporal GNNs trained to work under partial retrieval, plus lookup policies that decide what to fetch.
- **Compute:** AWS (EC2 GPU + S3) in `us-west-2`.

## Milestone (for graders)

- **Report:** [milestone/milestone.pdf](milestone/milestone.pdf) (LaTeX source: [milestone/milestone.tex](milestone/milestone.tex)).
- **Code version:** the git tag [`milestone`](https://github.com/MaxLuisRodriguez/CS224W-fraud-detection-machine-learning-with-graphs/tree/milestone) is exactly the code the report's numbers came from.
- **Where to start reading:** the code map below, in pipeline order. Every file opens with a plain-language summary of what it does and where it fits.
- **Tests:** `python -m pytest tests` (11 tests: metrics, temporal leakage, the lookup environment's guarantees, policy budgets).

## Code map

Everything below runs from the repo root after `pip install -e .`.
Datasets download automatically from RelBench on first use.

| Step | File | What it does |
|---|---|---|
| Data | [src/wtl/data.py](src/wtl/data.py) | Loads a RelBench database and task; builds the heterogeneous temporal graph (rows -> nodes, foreign keys -> typed edges, GloVe text features); temporal `NeighborLoader` |
| Model | [src/wtl/model.py](src/wtl/model.py) | Relational GNN: per-table encoders, time encoding, 2-layer heterogeneous GraphSAGE, linear head; `revealed` + `hide="drop"/"mask"` for partial retrieval |
| Partial view | [src/wtl/reveal.py](src/wtl/reveal.py) | Reveal a random share of rows, keyed on the original row id so shared rows are revealed together |
| Metrics | [src/wtl/metrics.py](src/wtl/metrics.py) | Precision@K, call-list metrics, per-list AUROC/AP/P@K, bootstrap intervals |
| Environment | [src/wtl/lookup.py](src/wtl/lookup.py) | A call list sampled once into one subgraph; `pull` and `enrich` lookups with a shared cache, list-before-get, and cost accounting |
| Policies | [src/wtl/policies.py](src/wtl/policies.py) | Random, case-local, uncertainty-first, coverage (batch-aware), sampled oracle-greedy (uses labels; a reference, not deployable) |
| Train / evaluate | [scripts/train_rdl.py](scripts/train_rdl.py) | Trains the GNN (full or random retrieval), evaluates at reveal shares p = 1 ... 0, saves metrics and weights |
| Baseline | [scripts/baseline_lightgbm.py](scripts/baseline_lightgbm.py) | LightGBM on each target's own row: the zero-lookup reference |
| Frontiers | [scripts/run_frontier.py](scripts/run_frontier.py) | Every policy on the same call lists and subgraphs, metrics at each budget |
| Analyses | [scripts/fig_frontier.py](scripts/fig_frontier.py), [scripts/table_value.py](scripts/table_value.py), [scripts/sharing_splits.py](scripts/sharing_splits.py), [scripts/summarize_runs.py](scripts/summarize_runs.py) | Frontier figure and table; value of each table; batch-aware advantage vs. sharing; seed averages |
| Checks | [scripts/check_reveal.py](scripts/check_reveal.py), [scripts/check_call_list.py](scripts/check_call_list.py), [scripts/check_lookup.py](scripts/check_lookup.py), [scripts/check_leakage.py](scripts/check_leakage.py), [scripts/check_checkpoint.py](scripts/check_checkpoint.py), [tests/](tests/) | Partial-view consistency, call-list structure, environment guarantees, temporal leakage on any dataset, checkpoint reload, policy budgets |
| Data exploration | [scripts/peek_data.py](scripts/peek_data.py), [scripts/dataset_card.py](scripts/dataset_card.py), [scripts/sharing_stats.py](scripts/sharing_stats.py), [scripts/inspect_graph.py](scripts/inspect_graph.py) | Dataset cards, sharing statistics, one sampled mini-batch |
| Report | [milestone/](milestone/) | Milestone report (LaTeX + PDF) and figures |

## Reproduce the milestone results

A GPU is recommended (the report used one NVIDIA L4); loading the `rel-event` graph needs about 40 GB of RAM including swap. Each command writes JSON results and logs under `outputs/`.

```bash
# E1/E2: train the GNN, full and random retrieval, three seeds each
for seed in 0 1 2; do for regime in full random; do
  python scripts/train_rdl.py --dataset rel-event --task user-ignore --train_reveal $regime --seed $seed
done; done
python scripts/summarize_runs.py --dataset rel-event --task user-ignore
python scripts/baseline_lightgbm.py --dataset rel-event --task user-ignore

# E3: budget frontiers (100 call lists, budgets up to 16 lookups per prospect) and figures
python scripts/run_frontier.py --dataset rel-event --task user-ignore --lists 100 --budgets 0.25 0.5 1 2 4 8 16 --name lists100
python scripts/fig_frontier.py --dataset rel-event --task user-ignore --name lists100 --metric auroc --ref "LightGBM, own row" 0.850
python scripts/fig_frontier.py --dataset rel-event --task user-ignore --name lists100 --metric p_at_k --ref "LightGBM, own row" 0.545

# E3 with the random-trained model, E4 table values, E5 sharing splits
python scripts/run_frontier.py --dataset rel-event --task user-ignore --run rdl_random_drop_ep20_seed0 --lists 30 --policies random case_local uncertainty coverage --name random_trained
python scripts/table_value.py --dataset rel-event --task user-ignore --lists 30
python scripts/sharing_splits.py --dataset rel-event --task user-ignore --name lists100

# Checks
python scripts/check_leakage.py --dataset rel-event --task user-ignore
python scripts/check_lookup.py --dataset rel-event --task user-ignore
python -m pytest tests
```

## Status

Milestone (Nov 2026): the relational GNN with partial retrieval, the lookup environment, five policies and the frontier runner are built and tested, and every milestone experiment ran on `rel-event` (see the report). Next: the learned lookup value, stronger batch-aware heuristics, and validity fixes listed in the report's next steps.
