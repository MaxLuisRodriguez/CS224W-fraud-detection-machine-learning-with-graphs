"""M7: the lookup environment's three guarantees.

1. Fetching everything gives exactly the full model's scores, so hiding rows is the only thing a budget changes.
2. Looking up something already in the cache is free.
3. List before get: an entity can only be fetched after a pulled row points to it.
"""
import pytest
import torch

from wtl.lookup import CallList, fetch_all


def test_full_retrieval_is_the_full_model(small_list):
    """With every row fetched, drop-mode scores must match the model run with nothing hidden.

    Why it matters: any gap at a smaller budget then comes from the hidden rows alone, not from the
    hiding machinery itself.
    """
    batch, entity, model = small_list
    cl = CallList(batch, entity)
    fetch_all(cl)
    assert all(m.all() for m in cl.masks().values())                # every sampled row fetched
    # The reference: the same model with no `revealed` dict, i.e. it sees every row.
    with torch.no_grad():
        full = torch.sigmoid(model(batch, entity))
    # allclose allows tiny floating-point differences.
    assert torch.allclose(cl.scores(model), full, atol=1e-5)


def test_cache_hits_are_free(small_list):
    """Repeating a pull or an enrich costs 0, so the budget only counts lookups that reveal something new."""
    batch, entity, _ = small_list
    cl = CallList(batch, entity)
    # The first legal pull reveals new rows (cost 1); the same pull again reveals nothing (cost 0).
    i, table = cl.legal_pulls()[0]
    assert cl.apply(("pull", i, table), i) == 1
    assert cl.apply(("pull", i, table), i) == 0                     # already pulled
    # That pull listed some entities; fetch the first one twice: 1, then 0.
    ent_table, (n_id, _, _) = next(iter(cl.enrich_candidates().items()))
    assert cl.apply(("enrich", ent_table, int(n_id[0]))) == 1
    assert cl.apply(("enrich", ent_table, int(n_id[0]))) == 0       # already in the cache
    # Two paid lookups in total.
    assert cl.cost == 2


def test_list_before_get(small_list):
    """Enriching an entity that no pulled row points to must raise ValueError.

    Why it matters: an agent can't ask for a record it doesn't know exists yet.
    """
    batch, entity, _ = small_list
    # Use a throwaway call list to find a real entity id: one that a pull would list.
    first = CallList(batch, entity)
    i, table = first.legal_pulls()[0]
    first.apply(("pull", i, table), i)
    ent_table, (n_id, _, _) = next(iter(first.enrich_candidates().items()))
    # On a fresh call list nothing is pulled, so fetching that entity directly is illegal.
    with pytest.raises(ValueError):
        CallList(batch, entity).apply(("enrich", ent_table, int(n_id[0])))   # nothing pulled yet points to it

        