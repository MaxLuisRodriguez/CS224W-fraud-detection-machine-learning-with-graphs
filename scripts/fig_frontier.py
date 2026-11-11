"""M9 (E3): the frontier figure and table.

Mean metric over call lists at each budget, one line per policy, with 95% bootstrap intervals; zero lookups and
full information as reference lines. Prints the same numbers as a markdown table.

What this answers:
    For a given number of lookups, which policy ranks the call list best, and how close does each policy
    get to full information?

How to run (from the repo root), after run_frontier.py:
    python scripts/fig_frontier.py --dataset rel-event --task user-ignore --name lists100 --metric auroc
    Add --ref "LightGBM, own row" <value> to draw an extra reference line (it can be repeated).

Reads outputs/<dataset>/<task>/frontier_<name>.json.
Writes the figure to --out (default milestone/figures/frontier_<dataset>_<metric>.pdf) and prints the table.

How to read the figure:
    x-axis: lookups per prospect (the budget), on a log2 scale, so each doubling takes the same width.
    y-axis: the chosen metric of one call list, averaged over call lists.
    Each colored line is one policy; its shaded band is a 95% bootstrap interval over call lists.
    Dotted gray line: zero lookups. Dashed gray line: full information (every lookup made).
    Dash-dot gray lines: any --ref values.
    A budget at or above a list's full cost is not run on that list, so high budgets can average over fewer
    lists; the table's "Lists" column says how many. A point needs at least two lists to be drawn.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wtl.metrics import bootstrap_ci

# Command-line options.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="rel-f1")
parser.add_argument("--task", default="driver-top3")
# Which frontier file to read: frontier_<name>.json.
parser.add_argument("--name", default="random_lists")
# Which metric to plot: AUROC, average precision, or precision@K (K = 10% of the list, rounded up).
parser.add_argument("--metric", default="auroc", choices=["auroc", "ap", "p_at_k"])
# Where to save the figure (default below). The file type follows the extension, e.g. .pdf or .png.
parser.add_argument("--out", default=None)
# extra reference lines, e.g. --ref "LightGBM, own row" 0.850 (a zero-lookup model scored on its own lists)
parser.add_argument("--ref", nargs=2, action="append", metavar=("LABEL", "VALUE"), default=[])
# name the --ref lines in the legend instead of beside the line (use when a label would overlap another one)
parser.add_argument("--ref_legend", action="store_true")
args = parser.parse_args()

# Load the per-list records written by run_frontier.py, and decide where the figure goes.
lists = json.loads((Path("outputs") / args.dataset / args.task / f"frontier_{args.name}.json").read_text())["lists"]
out = Path(args.out or f"milestone/figures/frontier_{args.dataset}_{args.metric}.pdf")

# Color and legend label for each policy; the dict order is also the drawing and legend order.
STYLE = {                                  # fixed order: a policy keeps its color in every figure
    "coverage": ("#2a78d6", "coverage (batch-aware)"),
    "case_local": ("#eb6834", "case-local"),
    "uncertainty": ("#1baf7a", "uncertainty-first"),
    "oracle_greedy": ("#eda100", "oracle-greedy (uses labels)"),
    "random": ("#898781", "random"),
}
# Colors for main text, secondary text and reference lines, and the grid.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def values(get) -> list:
    """One value per list, skipping lists where the metric is undefined.

    Args:
        get: a function that takes one list's record and returns a number, or None when there is none.
             (AUROC and AP are None on a list with only one class, and a policy may have no record at a
             budget that was skipped for that list.)

    Returns:
        The values that are not None, one per call list.
    """
    return [v for v in (get(L) for L in lists) if v is not None]


# Reference values: the mean over lists with zero lookups and with full information.
zero = values(lambda L: L["zero"][args.metric])
full = values(lambda L: L["full"][args.metric])
# One figure with one set of axes; figsize is (width, height) in inches, sized for a paper column.
fig, ax = plt.subplots(figsize=(5.5, 3.2))
# axhline draws a horizontal line across the whole plot: dotted for zero lookups, dashed for full information.
ax.axhline(np.mean(zero), color=MUTED, lw=1, ls=":")
ax.axhline(np.mean(full), color=MUTED, lw=1, ls="--")
# Rows for the markdown table, filled in while drawing.
table = []
# One line per policy, in STYLE order.
for name, (color, label) in STYLE.items():
    # Skip policies this frontier file did not run.
    if not any(name in L["policies"] for L in lists):
        continue
    # x positions (budgets), mean values, and the low and high ends of each interval.
    xs, mean, lo, hi = [], [], [], []
    # Every budget at which any list recorded this policy, smallest first.
    for b in sorted({r["budget"] for L in lists for r in L["policies"].get(name, [])}):
        # This policy's metric at budget b, one value per list (lists without a value are skipped).
        v = values(lambda L: next((r[args.metric] for r in L["policies"].get(name, []) if r["budget"] == b), None))
        # An interval needs at least two lists.
        if len(v) >= 2:
            # Bootstrap: resample the lists with replacement many times; the middle 95% of the resampled
            # means is the interval.
            ci = bootstrap_ci(v)
            xs.append(b)
            mean.append(np.mean(v))
            lo.append(ci[0])
            hi.append(ci[1])
            table.append((label, b, np.mean(v), ci, len(v)))
    # The shaded interval band (alpha = transparency, lw=0 = no outline), then the mean line with round
    # markers (mec/mew give each marker a white edge so overlapping markers stay readable).
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(xs, mean, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, label=label)

# Log2 x-axis, so 0.25, 0.5, 1, 2, 4 are evenly spaced.
ax.set_xscale("log", base=2)
# Put tick marks exactly at the plotted budgets, labeled like "0.25" and "2", with no small in-between ticks.
ticks = sorted({t[1] for t in table})
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.minorticks_off()
# Label the two reference lines at the left edge of the plot.
ax.text(ticks[0], np.mean(full), " full information", color=MUTED, fontsize=7, va="bottom")
ax.text(ticks[0], np.mean(zero), " zero lookups", color=MUTED, fontsize=7, va="bottom")
for label, value in args.ref:                    # optional extra reference lines (dash-dot)
    ax.axhline(float(value), color=MUTED, lw=1, ls="-.", label=label if args.ref_legend else None)
    if not args.ref_legend:
        ax.text(ticks[0], float(value), " " + label, color=MUTED, fontsize=7, va="top", ha="left")
# Axis labels and quiet styling: small fonts, a light horizontal grid behind the data, no top or right border.
ax.set_xlabel("lookups per prospect (budget)", fontsize=8, color=INK)
NAMES = {"auroc": "AUROC", "ap": "Average precision", "p_at_k": "Precision@13"}   # axis label per metric
ax.set_ylabel(f"{NAMES[args.metric]} per call list", fontsize=8, color=INK)
ax.tick_params(labelsize=7, colors=MUTED)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
# Legend below the plot in three columns; bbox_to_anchor places it under the x-axis label.
ax.legend(fontsize=7, frameon=False, labelcolor=INK, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
# Save; bbox_inches="tight" trims extra white space and keeps the legend below the plot inside the file.
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")

# The same numbers as a markdown table: one row per (policy, budget).
print(f"{args.dataset}: {args.metric}, mean over {len(lists)} call lists [95% bootstrap interval]\n")
print(f"zero lookups {np.mean(zero):.3f}   full information {np.mean(full):.3f}\n")
print("| Policy | Budget | Mean | 95% interval | Lists |")
print("|---|---:|---:|---|---:|")
for label, b, m, (l, h), n in table:
    print(f"| {label} | {b:g} | {m:.3f} | [{l:.3f}, {h:.3f}] | {n} |")
print("\nsaved", out)
