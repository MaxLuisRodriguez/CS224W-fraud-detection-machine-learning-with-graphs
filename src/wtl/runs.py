"""Rebuild a trained model exactly as it was trained, from the settings saved next to its weights.

train_rdl.py saves two files per run: <run>.json (its settings and results) and <run>.pt (its weights).
Every later experiment (frontiers, table values) must rebuild the SAME graph and the
SAME model before loading those weights: same dropped columns, same graph cache. This
module reads the settings from the JSON so no script has to repeat them by hand. Runs saved before these
settings existed simply get the defaults (every column, target-row time on).
"""
import json
from pathlib import Path

import torch

from wtl.data import build_graph, column_stypes, drop_columns, load
from wtl.model import RDLModel


def cache_name_for(drop_cols) -> str:
    """The graph-cache folder name for a set of dropped columns ("materialized" when nothing is dropped).

    Args:
        drop_cols: list of "table.column" names, or empty

    Returns:
        A folder name that differs whenever the set of dropped columns differs.
    """
    if not drop_cols:
        return "materialized"
    return "materialized_drop_" + "_".join(sorted(c.replace(".", "-") for c in drop_cols))


def graph_for(dataset: str, task_name: str, drop_cols=()):
    """Load a dataset and task and build its graph, optionally without some feature columns.

    Args:
        dataset:   RelBench dataset name, e.g. "rel-event"
        task_name: RelBench task name, e.g. "user-ignore"
        drop_cols: feature columns to leave out, each "table.column"

    Returns:
        task:      the RelBench task
        data:      the HeteroData graph
        col_stats: column statistics for the row encoders
    """
    _, task, db = load(dataset, task_name)
    stypes = column_stypes(db, dataset)
    if drop_cols:
        stypes = drop_columns(stypes, drop_cols)
    data, col_stats = build_graph(db, dataset, task, stypes, cache_name=cache_name_for(drop_cols))
    return task, data, col_stats


def run_settings(dataset: str, task_name: str, run: str) -> dict:
    """The settings a run was trained with, read from outputs/<dataset>/<task>/<run>.json ({} if missing)."""
    path = Path("outputs") / dataset / task_name / f"{run}.json"
    return json.loads(path.read_text())["args"] if path.exists() else {}


def load_run(dataset: str, task_name: str, run: str, device):
    """Rebuild the graph and the model a run was trained with, load its weights, and switch to eval mode.

    Args:
        dataset, task_name: which dataset and task
        run:    the run's file stem, e.g. "rdl_full_drop_ep20_seed0"
        device: torch device to put the model on

    Returns:
        task, data, model (model already on `device`, in eval mode)
    """
    cfg = run_settings(dataset, task_name, run)
    task, data, col_stats = graph_for(dataset, task_name, cfg.get("drop_cols") or [])
    model = RDLModel(data, col_stats, target_time=not cfg.get("no_target_time", False)).to(device)
    weights = Path("outputs") / dataset / task_name / f"{run}.pt"
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return task, data, model
