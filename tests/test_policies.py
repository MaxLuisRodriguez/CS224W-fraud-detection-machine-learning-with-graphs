"""M8: every policy spends exactly its budget, using legal lookups only.

The policy comparison is only fair if every policy gets the same number of paid lookups, and none of them
breaks the environment's rules (such as list before get).
"""
import numpy as np
import pytest

from wtl.lookup import CallList
from wtl.policies import POLICIES


# parametrize runs this test once per policy name, each reported as its own test.
@pytest.mark.parametrize("name", list(POLICIES))
def test_policy_spends_its_budget(small_list, name):
    """Run one policy until the budget is used up: it must never stall, cheat, or overspend."""
    batch, entity, model = small_list

    cl = CallList(batch, entity)
    rng = np.random.default_rng(0)

    # one lookup per prospect, on average
    budget = cl.n

    # Ask for small rounds of at most 4 lookups, never more than what is left. A lookup costs at most 1,
    # so the total can't go past the budget.
    while cl.cost < budget:
        picks = POLICIES[name](cl, min(4, budget - cl.cost), model, rng)
        assert picks, f"{name}, proposed nothing while lookups remain"
        for action, prospect in picks:
            cl.apply(action, prospect) # raises ValueError on an illegal lookup

    assert cl.cost == budget