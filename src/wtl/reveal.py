"""Which rows of a sampled mini-batch the model is allowed to see.

The output of this file is a "revealed" mask that RDLModel.forward takes: per table, one True/False per
sampled row. train_rdl.py uses random_reveal to train the model on partial views ("random-trained"),
and the check scripts use it to test that hiding rows behaves as intended.

Rows are revealed by their ORIGINAL row id (n_id), so every copy of a shared row is revealed together:
fetching a row once makes it visible to every target that touches it.
(Why copies exist: the time-aware loader samples a separate neighborhood for each target, so a row
linked to two targets shows up twice in the batch, once in each target's neighborhood.)
"""
import torch


#
## Random revelation: each distinct row is fetched with probability p
#
def random_reveal(batch, entity_table: str, p: float) -> dict:
    """Reveal each distinct row with probability p; the call-list targets are always revealed.

    This imitates an agent that has fetched a random share p of the database: p=0 shows only the
    targets themselves, p=1 shows everything.

    Args:
        batch: one HeteroData mini-batch from wtl.data.make_loader.
        entity_table: name of the table the targets come from (e.g. "users").
        p: probability that any one distinct row is revealed, between 0 and 1.

    Returns:
    {table name: bool tensor with one entry per sampled row (True = revealed)}
    """
    revealed = {}
    for node_type in batch.node_types:
        # n_id: for every sampled row in this table, its original row id in the full graph, shape [rows].
        n_id = batch[node_type].n_id
        # unique(): each original row once, so a row with several copies gets just one coin flip.
        distinct = n_id.unique()
        # torch.rand(k) gives k uniform numbers in [0, 1); "< p" turns each into True with chance p.
        coin = torch.rand(distinct.numel(), device=n_id.device) < p # one coin flip per distinct row
        chosen = distinct[coin]

        # torch.isin(a, b): for each entry of a, True if it appears anywhere in b.
        # So every copy of a chosen row is marked True together.
        revealed[node_type] = torch.isin(n_id, chosen)

    # The loader puts the targets first in the entity table, so its first num_targets rows are the
    # call list itself. Those are always visible: the agent knows who is on its list.
    num_targets = batch[entity_table].seed_time.size(0)

    # Note: only these target slots are forced on. If a target also appears deeper in another
    # target's neighborhood, that extra copy follows its coin flip like any other row.
    revealed[entity_table][:num_targets] = True
    return revealed