# Build modules

**Project type 1 · Real-world application of GNNs** (Medium blog post + Google Colab). Each module is a step-by-step guide: you type every line, and each step ends with a PASS check or a STOP rule.

## Proposal track: due Tue Oct 20, 11:59pm PT

One module per bullet of the handout's proposal instructions (§1.3), plus setup and assembly. Each module ends by writing its proposal section into `proposal/notes.md`, so P8 is assembly, not writing from scratch.

| # | Module | Handout bullet | Produces | Time | When |
|---|---|---|---|---|---|
| P0 | [Setup](P0-setup.md) | — | Repo skeleton, x64 environment, `check_env.py` | 45 min | Sat Oct 17 |
| P1 | [Application domain](P1-application-domain.md) | "Application domain" | Problem, RQ, H1–H5; proposal §1 | 30 min | Sat Oct 17 |
| P2 | [Datasets, and are they public?](P2-datasets-and-public.md) | "Which dataset(s)…?" "Is the dataset public?" | `peek_data.py`, `refs.bib`; §2 opening | 30 min | Sat Oct 17 |
| P3 | [Dataset, task, metric](P3-dataset-task-metric.md) | "Describe the dataset/task/metric" | Dataset cards, `metrics.py` + test; §2 | 1 h | Sat Oct 17 |
| P4 | [Why these datasets?](P4-why-these-datasets.md) | "Why did you choose the dataset?" | Headroom table, `sharing_stats.py`; §2 second half | 45 min | Sat Oct 17 |
| P5 | [Which graph ML model?](P5-which-model.md) | "Which graph ML model…? (Be sure to cite!)" | `text.py`, `data.py`, `model.py`, `inspect_graph.py`; §3 opening | 1 h | Sat Oct 17 |
| P6 | [Describe the model](P6-describe-the-model.md) | "Describe the model(s) (try using figures and equations)" | `reveal.py`, masked model, equations, Fig. 1 (+ optional Fig. 2); §3 | 1.5 h | Sat Oct 17 |
| P7 | [Why the model fits, and preliminary results](P7-why-appropriate.md) | "Why is/are the model(s) appropriate…?" | LightGBM baseline, leakage test, seed-averaged RDL results; §3 end, §4 | 2 h (mostly training in the background) | Sat Oct 17 |
| P8 | [Assemble and submit](P8-assemble-and-submit.md) | §4.1 format; "Who is in your group" | `proposal.pdf` | ~4 h | Sun Oct 18 draft · Mon Oct 19 polish · **Tue Oct 20 submit** |

P0–P7 add up to about 8 hours on Saturday. All of it runs on the laptop: `rel-f1` is the model dataset, and the two primary datasets are only read through manifests and DuckDB.

## Build track: after the proposal

| # | Module | Produces | Target dates |
|---|---|---|---|
| B1 | [Cloud GPU and full-information baselines](B1-cloud-gpu-and-baseline.md) | AWS guardrails, GPU instance, full-information RDL on `rel-event` and `rel-avito` (using P5–P7's code) | Oct 21 – Oct 25 |
| B2 | Lookup environment *(to be written)* | `list` / `get` actions with pull/get semantics keyed on `n_id`, one shared cache, budgets, list-before-get | Oct 26 – Nov 1 |
| B3 | Heuristic policies and frontiers *(to be written)* | Random, case-local, uncertainty, coverage, oracle-greedy; first budget frontiers (H1, H3, H4) | Nov 2 – Nov 9 |
| — | **Milestone** | Code + report draft | **Tue Nov 10, 11:59pm PT** |
| B4 | Learned batch value *(to be written)* | Learned batch-value policy, Zhou-style singleton baseline, ablations (H5) | Nov 11 – Nov 24 |
| B5 | Blog post and Colab *(to be written)* | Figures, rollout GIF, PyG snippets, documented Colab | Nov 25 – Dec 9 |
| — | **Final** | Private Medium draft link, account name, Colab | **Thu Dec 10, 11:59pm PT** |
