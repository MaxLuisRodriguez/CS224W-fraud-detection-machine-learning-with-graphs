# Results

Curated result files behind the numbers in the write-up. Every comparison is paired: all policies run on the same
sampled call lists with the same frozen scorer. A call list is 128 prospects from one prediction day; a budget `b`
is lookups per prospect (128 × b paid lookups per list); metrics are computed per list.

| Folder | Contents |
| --- | --- |
| `test_day/<dataset>.json` | The pre-registered test day: 100 call lists per dataset. Per list: full-retrieval cost, share of listed entities shared by two or more prospects, AUROC / AP / Precision@13 with zero lookups and with full information, and for every policy one record per budget (`budget`, `cost`, `auroc`, `ap`, `p_at_k`). `prospects` holds each prospect's label and its zero- and full-information scores (used for calibration). |
| `test_replication/<dataset>.json` | The same 100 test lists rerun with three value-model seeds and baselines adapted from active feature acquisition (`learned_*_batch_v3`: label-supervised value; `learned_knn_batch_v3`: nearest-neighbour value; `inst_*`: equal budget per prospect). Records also hold `plan_s`, the seconds spent choosing lookups. |
| `scorer_seeds/<dataset>.json` | The whole pipeline repeated with scorer seeds 1 and 2 (scorer, value-model training data, value model, test run): heuristics and `learned_sage_eig_v3sc<seed>`. |
| `validation_scores/<dataset>.json` | Labels and full-information scores on 20 validation lists (for calibration on validation days). |
| `tables/*.csv` | Every results table, as reported. |
| `synthetic/summary.json` | The synthetic grid varying entity sharing (0%, 18%, 49%) and whether labels depend on wiring. |

Policy names: `random`, `case_local`, `uncertainty` (uncertainty-first), `coverage`, `breadth_then_share`, the
`_ordered` variants (tables pulled in an order measured on training days), and `learned_<model>_<target>_v3`, where
`<model>` is the value model (`sage`, `gat`, `transformer` with message passing; `mlp`, `lgbm`, `knn` without) and
`<target>` is `eig` (expected information gain, no labels) or `batch` (realized loss reduction, label-supervised).
References that use information a real agent lacks: `oracle_greedy` (labels) and `eig_greedy` (each lookup's
result); `_all` scores every candidate instead of 64 sampled ones.

Datasets and tasks come from RelBench: rel-event/user-ignore, rel-stack/user-engagement, rel-trial/study-outcome,
rel-avito/user-clicks, rel-hm/user-churn and rel-amazon/user-churn.
