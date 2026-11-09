"""M9 (E5): does the batch-aware advantage grow with sharing?

Reads a frontier file, gives every call list one number (coverage's AUROC minus case-local's, averaged over
budgets up to --max_budget), splits the lists into thirds by how much they share, and compares the thirds.

Why ask this:
    The coverage policy plans for the whole list at once: it prefers lookups that inform many prospects,
    such as an event several prospects attend. The case-local policy researches each prospect on its own.
    Planning for the whole list should matter most when prospects actually share entities. So if that is
    why coverage does better, its advantage should be larger on lists that share more.

"Sharing" of a list = the share of listed entities that two or more prospects point to (computed by
run_frontier.py with every lookup made).

How to run (from the repo root), after run_frontier.py:
    python scripts/sharing_splits.py --dataset rel-event --task user-ignore --name lists100

Reads outputs/<dataset>/<task>/frontier_<name>.json. Writes nothing; it prints a markdown table.

How to read the table:
    One row per third of the lists, from least to most shared. "Shared entities" is the range of sharing in
    that third. "Advantage" is the mean AUROC gap (positive = --a ranks better than --b). The 95% interval
    comes from resampling the lists in that third. An advantage that grows from the first row to the last
    supports the idea; wide or overlapping intervals mean the evidence is weak.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wtl.metrics import bootstrap_ci

# Command-line options.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
# Which frontier file to read: frontier_<name>.json.
parser.add_argument("--name", default="random_lists")
parser.add_argument("--a", default="coverage")              # the batch-aware policy
parser.add_argument("--b", default="case_local")            # the per-prospect policy
# Only budgets up to this one count. At large budgets both policies end up buying most of the same
# lookups, so the comparison is most telling at small budgets.
parser.add_argument("--max_budget", type=float, default=2.0)
args = parser.parse_args()

# Load the per-list records written by run_frontier.py.
path = Path("outputs") / args.dataset / args.task / f"frontier_{args.name}.json"
lists = json.loads(path.read_text())["lists"]

# Step 1: one number per list: the average AUROC gap between the two policies.
per_list = []
for L in lists:
    # budget -> AUROC, for each of the two policies on this list.
    a = {r["budget"]: r["auroc"] for r in L["policies"][args.a]}
    b = {r["budget"]: r["auroc"] for r in L["policies"][args.b]}
    # Gaps at budgets both policies reached, up to --max_budget, where AUROC is defined (not None).
    gaps = [a[x] - b[x] for x in a if x in b and x <= args.max_budget and a[x] is not None and b[x] is not None]
    # A list with no usable budget is skipped. Each kept list becomes a pair (sharing, mean gap).
    if gaps:
        per_list.append((L["shared"], float(np.mean(gaps))))
per_list.sort()                                              # least-shared lists first

# Step 2: split the sorted lists into three nearly equal groups and summarize each one.
print(f"{args.dataset}: {args.a} minus {args.b}, AUROC averaged over budgets <= {args.max_budget}; {len(per_list)} lists\n")
print("| Lists | Shared entities | Advantage | 95% interval |")
print("|---|---|---:|---|")
# np.array_split turns the (sharing, gap) pairs into a 2-column array and cuts it into 3 parts. When the
# count does not divide by 3, the first parts get one extra list each.
for label, part in zip(["least shared third", "middle third", "most shared third"], np.array_split(per_list, 3)):
    # With fewer than 3 lists, a third can be empty; skip it.
    if len(part) == 0:
        continue
    # 95% interval for this third's mean gap, by resampling its lists with replacement.
    lo, hi = bootstrap_ci(part[:, 1])
    # The lists are sorted by sharing, so the first and last rows give this third's lowest and highest sharing.
    print(f"| {label} | {part[0, 0]:.0%} to {part[-1, 0]:.0%} | {part[:, 1].mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
