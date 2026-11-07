"""Which rows of a sampled mini-batch the model is allowed to see.

Rows are revealed by their ORIGINAL row id (n_id), so every copy of a shared row is revealed together:
fetching a row once makes it visible to every target that touches it.
"""
import torch


#
## Random revelation: each distinct  row is fetched with probability p
#
def random_reveal(batch, entity_table: str, p: float) -> dict:
    """Reveal each distinct row with probability p; the call-list targets are always revealed.
    
    Returns:
    {table name: bool tensor with one entry per sampled row (True = revealed)}
    """
    revealed = {}
    for node_type in batch.node_types:
        n_id = batch[node_type].n_id
        distinct = n_id.unique()
        coin = torch.rand(distinct.numel(), device=n_id.device) < p # one coin flip per distinct row
        chosen = distinct[coin]

        revealed[node_type] = torch.isin(n_id, chosen)

    num_targets = batch[entity_table].seed_time.size(0)

    revealed[entity_table][:num_targets] = True
    return revealed