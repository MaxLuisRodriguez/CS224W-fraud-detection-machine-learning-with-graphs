"""M9: retrieval-budget frontiers.

What this answers:
    A research agent has a call list of N prospects and can only afford so many lookups. For each lookup
    policy (a rule for choosing the next lookups), how good is the list's ranking after it has spent a given
    budget? Ranking quality plotted against budget is that policy's "frontier".

For each call list: sample it once, then run every policy on that same subgraph with the same trained model,
recording ranking quality each time the spend reaches a budget (lookups per prospect).

Key ideas:
    - A budget b means b lookups per prospect on average, so b * N paid lookups for the whole list.
      A row fetched once is visible to every prospect that touches it, and fetching it again is free
      (a cache hit), so one lookup can help many prospects.
    - Paired design: every policy starts from nothing on the SAME sampled subgraph, with the same model and
      the same random seed, so differences between policies do not come from different samples.
    - Policies re-plan in rounds of N/4 lookups, so later picks can react to what earlier picks revealed.
    - Two reference points per list: "zero" (no lookups: only the prospects' own rows) and "full" (every
      possible lookup made). "full_cost" is how many lookups full information takes.
    - Budgets at or above a list's full cost are skipped for that list (there is nothing left to buy).
    - oracle_greedy peeks at the labels and scores up to 64 candidate lookups, one forward pass each, every
      round. That is slow, so it is capped at --oracle_max_budget.
    - Run every policy in ONE invocation. The neighbor sampler is not reproducible across runs, so policies
      are only compared fairly when they share the same sampled subgraphs, which happens within one run.

How to run (from the repo root), after train_rdl.py has saved weights:
    python scripts/run_frontier.py --dataset rel-event --task user-ignore --lists 100 --name lists100

Reads outputs/<dataset>/<task>/<run>.pt (the frozen trained weights).
Writes outputs/<dataset>/<task>/frontier_<name>.json, updated after every list.
    It holds "args" and one record per list: seed, one_day, full_cost, shared, the "zero" and "full"
    metrics, and for each policy one entry per budget (budget, cost actually spent, auroc, ap, p_at_k, and
    paid lookups per table in "fetched").

How to read the printed lines (one per list):
    the full cost (and per prospect); "shared" = the share of listed entities that two or more prospects
    point to; AUROC with zero lookups and with full information; each policy's AUROC at the highest budget
    it reached on this list; and the seconds the list took. fig_frontier.py and sharing_splits.py turn the
    JSON into the figure and tables.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from wtl.runs import load_run
from wtl.lookup import CallList, fetch_all, sample_call_list
from wtl.model import RDLModel
from wtl.policies import POLICIES

def run_policy(cl, policy, budgets, model, rng, round_size: int) -> list:
    """Spend up to each budget in turn (re-planning every round_size lookups); one record per budget.

    The spend is cumulative: lookups bought for budget 0.5 are still there at budget 1. So the records trace
    one policy's progress as it keeps spending, like an agent that could be stopped at any point.

    Args:
        cl: a fresh CallList (nothing fetched yet). It is changed in place as lookups are made.
        policy: a function policy(cl, k, model, rng) that proposes up to k (action, prospect) lookups.
        budgets: lookups per prospect at which to stop and score, in increasing order.
        model: the frozen trained model, used to score the list (and by policies that look at scores).
        rng: numpy random generator, for policies that make random choices.
        round_size: the most lookups taken before the policy is asked to plan again.

    Returns:
        A list with one dict per budget: the budget, the cost actually spent, the list's metrics (auroc, ap,
        p_at_k) and the number of paid lookups per table.
    """

    records = []
    for b in budgets:
        # Total paid lookups allowed at this budget: b per prospect times N prospects.
        target = round(b * cl.n)
        # Keep asking the policy for lookups until the spend reaches the target.
        while cl.cost < target:
            # Ask for one round, but never more than what is left of the budget.
            picks = policy(cl, min(round_size, target - cl.cost), model, rng)
            # The policy has nothing left to propose: stop early.
            if not picks:
                break

            # Make each lookup. It costs 1, or 0 if everything it reveals is already cached, and is charged
            # to the named prospect (None = it serves the whole list).
            for action, prospect in picks:
                cl.apply(action, prospect)

        # Score the list as it stands. "cost" can be below the target if the policy ran out of lookups.
        # dict(...) copies the per-table counts, so later spending does not change this record.
        records.append({"budget": b, "cost": cl.cost, **cl.metrics(model), "fetched": dict(cl.fetched)})

    return records

def main():
    """Load the frozen model, then build every policy's frontier on each call list, saving after every list."""
    # Command-line options.
    parser = argparse.ArgumentParser()
    # Which RelBench database and task.
    parser.add_argument("--dataset", default="rel-f1")
    parser.add_argument("--task", default="driver-top3")
    # Which saved training run to load: the name of a .pt file written by train_rdl.py, without ".pt".
    parser.add_argument("--run", default="rdl_full_drop_ep20_seed0")
    # How many call lists to sample (seeds 0, 1, 2, ...), and how many prospects per list (N).
    parser.add_argument("--lists", type=int, default=30)
    parser.add_argument("--size", type=int, default=128)
    # Which policies to run (default: every policy in wtl.policies.POLICIES).
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    # Budgets in lookups per prospect; the list is scored when the spend reaches each one.
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
    # The highest budget oracle_greedy is run to (it needs many forward passes per round).
    parser.add_argument("--oracle_max_budget", type=float, default=1.0)
    # Name for the output file: frontier_<name>.json.
    parser.add_argument("--name", default="random_lists")
    args = parser.parse_args()

    # Step 1: load the data, build the graph, and load the frozen trained model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_run (wtl/runs.py) rebuilds the graph and the model exactly as the run was trained (same dropped
    # columns, same target-row setting), loads the trained weights, and switches to evaluation mode. The
    # weights are never updated here.
    task, data, model = load_run(args.dataset, args.task, args.run, device)
    # The table whose rows are the prospects (users on rel-event).
    entity = task.entity_table
    out_dir = Path("outputs") / args.dataset / args.task

    # Everything goes into one dict, which is written to disk after each list.
    results = {"args": vars(args), "lists": []}
    out = out_dir / f"frontier_{args.name}.json"
    # Step 2: one call list per seed.
    for seed in range(args.lists):
        start = time.time()
        # Draw N prospects scored on the same day (when the data allows it) and sample all their 2-hop
        # neighborhoods into ONE batch. Every policy below reuses this exact batch.
        # (rows = the task rows drawn, unused here; one_day = whether all prospects share one day.)
        batch, rows, one_day = sample_call_list(data, task, "val", args.size, seed)
        batch = batch.to(device)

        # Full information: make every possible lookup on one CallList. fetch_all also returns "shared",
        # the share of listed entities that two or more prospects point to (how much lookups can be shared).
        full = CallList(batch, entity)
        shared = fetch_all(full)

        # The per-list record. A fresh CallList has fetched nothing beyond the prospects' own rows, so its
        # metrics are the zero-lookup reference; "full" is the full-information reference.
        record = {
            "seed": seed,
            "one_day": one_day,
            "full_cost": full.cost,
            "shared": shared,
            "zero": CallList(batch, entity).metrics(model), "full": full.metrics(model),
            "policies": {}
        }

        # Step 3: run each policy on its own fresh CallList over the SAME batch (a paired comparison).
        for name in args.policies: 
            # oracle_greedy stops at --oracle_max_budget; the other policies have no cap.
            cap = args.oracle_max_budget if name == "oracle_greedy" else float("inf")
            # Keep only budgets below this list's full cost (at or above it, every policy would simply
            # reach full information) and within the cap.
            budgets = [b for b in args.budgets if b * args.size < full.cost and b <= cap]

            # Start from nothing, and give every policy the same seeded random generator.
            cl = CallList(batch, entity)
            rng = np.random.default_rng(seed)

            # Re-plan every N/4 lookups (at least 1).
            record["policies"][name] = run_policy(cl, POLICIES[name], budgets, model, rng, max(1, args.size // 4))

        # Save after every list, so a crash or a stopped cloud machine loses at most the current list.
        results["lists"].append(record)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
        # One progress line. Each policy's AUROC at its highest budget for this list; a policy is left out if
        # it recorded nothing or its AUROC is undefined (a list with only one class has no AUROC).
        line = "  ".join(f"{n} {r[-1]['auroc']:.3f}" for n, r in record["policies"].items() if r and r[-1]["auroc"])
        print(f"list {seed}: full cost {full.cost} ({full.cost / args.size:.1f}/prospect), shared {shared:.0%}, "
              f"zero {record['zero']['auroc']:.3f}, full {record['full']['auroc']:.3f} | at top budget: {line} "
              f"| {time.time() - start:.0f} s", flush=True)
    print("saved", out)


# Run main() only when this file is executed as a script, not when it is imported.
if __name__ == "__main__":
    main()