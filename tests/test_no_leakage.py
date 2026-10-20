"""P7: every sampled row is dated no later than the prediction time of the target it was sampled for."""
import pytest

from wtl.data import build_graph, column_stypes, load, make_loader


@pytest.fixture(scope="module")
def batch_and_task():
    """Build the rel-f1 graph once, and hand one validation batch of 64 targets to every test."""
    dataset, task, db = load("rel-f1", "driver-top3")

    data, _ = build_graph(db, "rel-f1", task, column_stypes(db, "rel-f1"))
    loader, _ = make_loader(data, task, "val", batch_size=64)
    return next(iter(loader)), task


def test_no_rows_from_the_future(batch_and_task):
    batch, task = batch_and_task
    seed_time = batch[task.entity_table].seed_time  # each target's prediction time

    checked = 0

    for node_type in batch.node_types:
        store = batch[node_type]
        if "time" not in store or store.num_nodes == 0:
            continue # tables without dates can't leak the future

        owner = store.batch
        assert (store.time <= seed_time[owner]).all(), f"{node_type} contains rows after the prediction time"
        checked += 1
    assert checked > 0, "no timestamped tables were checked"