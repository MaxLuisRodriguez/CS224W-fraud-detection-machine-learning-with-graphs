"""P7: every sampled row is dated no later than the prediction time of the target it was sampled for.

This is the no-leakage check. If the sampler let in rows from after the prediction time, the model could
peek at the future, and its scores would look better than they could be in real use.
"""
import pytest

from wtl.data import build_graph, column_stypes, load, make_loader


# scope="module": build the graph once for this file, since that is the slow part.
@pytest.fixture(scope="module")
def batch_and_task():
    """Build the rel-f1 graph once, and hand one validation batch of 64 targets to every test."""
    dataset, task, db = load("rel-f1", "driver-top3")

    data, _ = build_graph(db, "rel-f1", task, column_stypes(db, "rel-f1"))
    loader, _ = make_loader(data, task, "val", batch_size=64)
    return next(iter(loader)), task


def test_no_rows_from_the_future(batch_and_task):
    """For every timestamped table, no sampled row may be dated after its own target's prediction time."""
    batch, task = batch_and_task
    seed_time = batch[task.entity_table].seed_time  # each target's prediction time

    checked = 0

    for node_type in batch.node_types:
        store = batch[node_type]
        if "time" not in store or store.num_nodes == 0:
            continue # tables without dates can't leak the future

        # Sampling is disjoint per target, so store.batch gives, for each sampled row, the index of the target
        # it was sampled for. seed_time[owner] then lines up that target's prediction time with each row.
        owner = store.batch
        assert (store.time <= seed_time[owner]).all(), f"{node_type} contains rows after the prediction time"
        checked += 1
    # Guard against a test that passes by checking nothing.
    assert checked > 0, "no timestamped tables were checked"