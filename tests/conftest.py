"""Shared test fixtures. pytest loads this file automatically for every test in this folder.

A fixture is a piece of setup that pytest builds and hands to any test that names it as an argument.
The one here builds a small call list on rel-f1, the small development dataset, so the lookup and policy
tests run quickly.
"""
import pytest
import torch

from wtl.data import build_graph, column_stypes, load
from wtl.lookup import sample_call_list
from wtl.model import RDLModel


# scope="module": built once per test file and shared by that file's tests, since building the graph is slow.
@pytest.fixture(scope="module")
def small_list():
    """One 16-prospect rel-f1 call list and an untrained model: the guarantees tested don't depend on training.

    Returns:
        batch:  the call list's sampled HeteroData batch
        entity: the prospects' table name (rel-f1: "drivers")
        model:  an untrained RDLModel in eval mode
    """
    # Fix the random seed so the untrained model's random weights are the same on every run.
    torch.manual_seed(0)
    # Load rel-f1 and turn its database into one graph (rows -> nodes, foreign keys -> edges).
    dataset, task, db = load("rel-f1", "driver-top3")
    data, col_stats = build_graph(db, "rel-f1", task, column_stypes(db, "rel-f1"))

    # Draw 16 validation prospects and sample their neighborhoods into one batch.
    batch, _, _ = sample_call_list(data, task, "val", size=16, seed=0)
    # A fresh model; eval() puts any layers that act differently during training into inference mode.
    model = RDLModel(data, col_stats)
    model.eval()

    return batch, task.entity_table, model
