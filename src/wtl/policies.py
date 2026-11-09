"""Lookup policies (M8). Each one proposes the next k lookups for a call list; the frontier runner applies them.

A policy is a rule for spending a lookup budget: given what has been fetched so far, which lookups come next?
The policies differ in what they look at: nothing (random), one prospect at a time (case_local,
uncertainty_first), the whole list at once (coverage), or the true labels (oracle_greedy, a reference only).

Every policy has the same signature:   policy(cl, k, model, rng) -> [(action, prospect), ...]
  action   = ("pull", prospect, table) or ("enrich", table, n_id)
  prospect = whose budget the lookup is charged to (None when it serves the whole list)
"""
import numpy as np
import torch
import torch.nn.functional as F

def all_actions(cl) -> list:
    """Every legal lookup right now: each pending pull, and each listed entity once.

    Args:
        cl: the CallList whose state we read

    Returns:
        list of (action, prospect); pulls are charged to their prospect, enrichments to None
    """
    # Every (prospect, table) pair not pulled yet.
    actions = [(("pull", i, t), i) for i, t in cl.legal_pulls()]

    # Every listed entity. An entity several prospects point to appears once per prospect in the candidates,
    # so unique() keeps one copy of each.
    for table, (n_id, prospect, mentions) in cl.enrich_candidates().items():
        actions += [(("enrich", table, nid), None) for nid in n_id.unique().tolist()]
    return actions

def next_for(cl, i: int, cands: dict):
    """Prospect i's own next lookup: its next table to pull, else the entity its pulled rows mention most.

    Args:
        cl:    the CallList
        i:     the prospect's position on the list
        cands: cl.enrich_candidates(), computed once by the caller

    Returns:
        one action, or None if prospect i has nothing left to look up
    """
    # Pulls first. legal_pulls() lists each prospect's tables in alphabetical order, so this takes the
    # alphabetically first table prospect i hasn't pulled yet.
    for j, t in cl.legal_pulls():
        if j==i:
            return ("pull", i, t)

    # All pulled: pick the entity that prospect i's rows mention most, across all tables.
    best, best_mentions = None, 0

    for table, (n_id, prospect, mentions) in cands.items():
        # mine marks the candidate entries that belong to prospect i.
        mine = prospect == i
        if mine.any():
            # Multiplying by the bool mask zeroes the other prospects' counts, so argmax lands on one of
            # prospect i's entities (the first one on a tie).
            top = int(torch.argmax(mentions * mine))
            # A later table wins only with strictly more mentions.
            if int(mentions[top]) > best_mentions:
                best, best_mentions = ("enrich", table, int(n_id[top])), int(mentions[top])

    return best


def by_prospect(cl, order, k: int) -> list:
    """Walk prospects in `order`; each takes its own next lookup, until k lookups are proposed.

    Args:
        cl:    the CallList
        order: prospect positions, most deserving first
        k:     how many lookups to propose

    Returns:
        up to k (action, prospect) pairs, at most one per prospect
    """

    # Compute the candidates once: the state doesn't change while we are only proposing.
    cands = cl.enrich_candidates()
    out = []

    # Each prospect in turn proposes its next lookup; prospects with nothing left are skipped.
    for i in order:
        action = next_for(cl,int(i), cands)
        if action is not None:
            out.append((action, int(i)))
        if len(out) == k:
            break

    return out


# =====================================================================================================
# The heuristics
# =====================================================================================================
def random_policy(cl, k, model, rng) -> list:
    """The floor: k legal lookups uniformly at random."""
    # Choose k distinct positions in the list of legal lookups (fewer if fewer exist).
    actions = all_actions(cl)
    pick = rng.choice(len(actions), size=min(k, len(actions)), replace=False)
    return [actions[j] for j in pick]

def case_local(cl, k, model, rng) -> list:
    """Research each prospect on its own: the least-researched prospects each take their next lookup."""

    # argsort of the per-prospect spend puts the least-researched prospects first. kind="stable" keeps
    # prospects with equal spend in list order, so ties are broken the same way every time.
    return by_prospect(cl, np.argsort(cl.spent, kind="stable"), k)

def uncertainty_first(cl, k, model, rng) -> list:
    """Active feature acquisition: research the prospects whose score is closest to 0.5 first.

    Active feature acquisition means buying extra inputs for the cases the model is least sure about.
    """

    # p * (1 - p) peaks at p = 0.5 (most uncertain) and falls to 0 at p = 0 or 1. Sorting its negative puts the
    # most uncertain prospects first; kind="stable" keeps ties in list order.
    p = cl.scores(model).cpu().numpy()
    return by_prospect(cl, np.argsort(-(p * (1-p)), kind="stable"), k)

def coverage(cl, k, model, rng) -> list:
    """Batch-aware: take the lookups that inform the most prospects.

    An entity informs every prospect whose pulled rows point to it; a pull informs one prospect. Ties go to
    pulls (they list new entities), and among pulls to the least-researched prospect.
    """
    # Score every legal lookup as a tuple (reach, kind, spent, action, prospect):
    #   reach = prospects informed; kind = 0 for a pull, 1 for an enrich; spent = the prospect's spend so far.
    scored = [(1, 0, cl.spent[i], ("pull", i, t), i) for i, t in cl.legal_pulls()]
    for table, (n_id, prospect, mentions) in cl.enrich_candidates().items():
        # Each candidate entry is one (entity, prospect) pair, so counting an entity's entries counts
        # the prospects it informs.
        ent, reach = n_id.unique(return_counts=True)                  # prospects per entity
        scored += [(int(r), 1, 0, ("enrich", table, int(e)), None) for e, r in zip(ent, reach)]
    # Sort by most prospects informed, then pulls before enrichments, then least-researched prospect.
    # Python's sort is stable, so any remaining ties keep the order above.
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    return [(s[3], s[4]) for s in scored[:k]]

#
# The label-informed reference (uses the labels; never deployable; not a guaranteed upper bound)
#
def list_loss(p: torch.Tensor, y: torch.Tensor) -> float:
    """Total binary cross-entropy over the list: the quantity the batch value (proposal Eq. 5) reduces.

    Args:
        p: shape [N], predicted probabilities
        y: shape [N], 0/1 labels as floats

    Returns:
        the summed loss as a Python float (lower = the probabilities sit closer to the true labels)
    """
    # Clamp keeps probabilities away from exactly 0 or 1, where the log in the loss would be infinite.
    return F.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), y, reduction="sum").item()

def oracle_greedy(cl, k, model, rng, n_candidates: int = 64) -> list:
    """Among a random sample of legal lookups, take the k with the largest true drop in list loss.

    It peeks at the labels, so it is a reference point, not a policy a real agent could run. Each lookup is
    scored on its own (one step ahead), so it is not guaranteed to be the best possible choice of k lookups.
    """
    # Step 1: sample up to n_candidates legal lookups (scoring all of them would be slow).
    actions = all_actions(cl)
    sample = rng.choice(len(actions), size=min(n_candidates, len(actions)), replace=False)
    # Step 2: the batch value of each sampled lookup = loss now minus loss if that one lookup were added.
    # scores(model, extra=...) is a what-if: it does not change the call list's state.
    base = list_loss(cl.scores(model), cl.y)
    gains = [base - list_loss(cl.scores(model, extra=actions[j][0]), cl.y) for j in sample]
    # Step 3: take the k largest gains (argsort is ascending, so reverse it).
    best = np.argsort(gains)[::-1][:k]
    return [actions[sample[j]] for j in best]


# The names the frontier runner and the tests use to look up each policy.
POLICIES = {
    "random": random_policy,
    "case_local": case_local,
    "uncertainty": uncertainty_first,
    "coverage": coverage,
    "oracle_greedy": oracle_greedy,
}

