"""Ranking metrics for call lists.

A sales rep can only make the first K calls of a list, so these metrics score how good the TOP of a
ranking is, on lists the size of a real call list, not just the whole dataset at once.

Who uses what: train_rdl.py and the baselines call call_list_metrics (many random lists at once);
the lookup environment (wtl.lookup) calls list_metrics to score its one call list as it stands; the
result and figure scripts call bootstrap_ci to put a 95% interval around an average over lists.
"""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

#
# Precision@k: "Of the K calls I make first, how many were the right ones?"
#
def precision_at_k(y_true, scores, k:int) -> float:
    """Fraction of real positives among the k highest-scored prospects.

    Args:
        y_true: 0/1 labels, one per prospect (1 = the outcome we want to find).
        scores: model scores, one per prospect; higher = call earlier.
        k: how many calls the rep makes from the top of the ranking.

    Returns:
        A number in [0,1]. Random scores give about the positive rate; a perfect ranking gives
        min(1, positives_in_list / k).

    Example:
    >>> precision_at_k([1, 0, 1, 0, 0], [0.9, 0.1, 0.8, 0.3, 0.2], k=2)
    1.0
    """
    # np.asarray: accept lists, pandas columns or tensors-turned-arrays alike, as NumPy arrays.
    y_true, scores = np.asarray(y_true), np.asarray(scores)

    # np.argsort returns the positions that would sort the array from smallest to largest, so we
    # sort the negated scores to put the highest score first, then keep the first k positions.
    top = np.argsort(-scores, kind="stable")[:k] # indices of the k highest scores (stable: ties keep input order)

    # Mean of 0/1 labels = share of positives among those k calls.
    return float(y_true[top].mean())


#
# Call-list metrics: AUROC, average precision, and precision@K, averaged over random call lists
#
def call_list_metrics(y_true, scores, list_size: int = 128, num_lists: int = 200, seed: int = 0) -> dict:
    """Score a model the way a rep experiences it: on many random call lists of `list_size` prospects.

    For each of `num_lists` random lists (drawn without replacement from all scored prospects, so no
    prospect appears twice within one list; different lists may share prospects):
    * AUROC: chance a random positive in the list is ranked above a random negative (0.5 = random).
    * Average precision (AP): how concentrated the positives are at the top (random = positive rate).
    * Precision@K with K = ceil(0.1 * list_size): of the first K calls, the share that were positives.
    Returns the average of each over the lists.

    AUROC and AP are undefined for a list with no positives (or no negatives), so those lists are
    skipped for those two metrics only; "lists_scored" reports how many counted.

    Args:
        y_true: 0/1 labels for every scored prospect (e.g. the whole validation day).
        scores: model scores, aligned with y_true.
        list_size: prospects per call list (N). Default 128.
        num_lists: how many random lists to average over. Default 200.
        seed: fixes which lists are drawn, so every model is scored on the same lists.

    Returns:
        {"auroc": float, "ap": float, "p@K": float, "lists_scored": int}, e.g. "p@13" for N = 128.
    """
    rng = np.random.default_rng(seed) # same seed -> same lists -> fair comparison between models
    y_true, scores = np.asarray(y_true), np.asarray(scores)

    k = math.ceil(0.1 * list_size) # the calls a rep makes first: 13 for a list of 128

    auroc, ap, p_at_k = [], [], []

    for _ in range(num_lists):
        # rng.choice(n, size, replace=False): list_size different positions out of 0..n-1.
        idx = rng.choice(len(y_true), size=list_size, replace=False) # one random call list
        y, s = y_true[idx], scores[idx]

        p_at_k.append(precision_at_k(y,s,k)) # always defined

        # AUROC and AP need at least one positive and one negative in the list.
        # roc_auc_score / average_precision_score (sklearn): take labels and scores, return one number.
        if 0 < y.sum() < len(y):
            auroc.append(roc_auc_score(y, s))
            ap.append(average_precision_score(y, s))

    # Average over the lists. If no list had both classes, np.mean of an empty list gives nan.
    return {"auroc": float(np.mean(auroc)), "ap": float(np.mean(ap)), f"p@{k}": float(np.mean(p_at_k)),
            "lists_scored": len(auroc)}


#
# One call list at one budget, and confidence intervals over lists (M9)
#

def list_metrics(y_true, scores) -> dict:
    """AUROC, AP and P@K (K = ceil(0.1 N)) of ONE call list. AUROC and AP are None if the list has one class.

    The lookup environment calls this to score its own call list with whatever rows have been
    fetched so far, so a run can track how the ranking changes as the budget is spent.

    Args:
        y_true: 0/1 labels, one per prospect on the list (N prospects).
        scores: model scores for the same prospects, in the same order.

    Returns:
        {"auroc": float or None, "ap": float or None, "p_at_k": float}. Note the key is "p_at_k"
        here, not "p@13" as in call_list_metrics.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    # True only if the list has at least one positive and at least one negative.
    both = 0 < y_true.sum() < len(y_true)

    return {
        "auroc": float(roc_auc_score(y_true, scores)) if both else None,
        "ap": float(average_precision_score(y_true, scores)) if both else None,
        "p_at_k": precision_at_k(y_true, scores, math.ceil(0.1 * len(y_true))),
    }


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0) -> tuple:
    """95% interval for the mean of `values` (one value per call list), by resampling lists with replacement.

    The idea (a "percentile bootstrap"): pretend the lists we have are the whole world, draw many fake
    sets of lists from them, and see how much the average moves. The middle 95% of those averages is
    the interval.

    Args:
        values: one number per call list (e.g. one policy's AUROC on each list at one budget).
        n_boot: how many resampled sets to draw. Default 2000.
        seed: fixes the random draws, so the same values always give the same interval.

    Returns:
        (low, high): the 2.5th and 97.5th percentiles of the resampled means.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    # rng.choice(values, size=(n_boot, n), replace=True): an [n_boot, n] table; each row is one
    # resampled set of n lists (some lists picked twice, some not at all). .mean(axis=1) averages
    # each row, giving n_boot means.
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)

    # np.percentile(a, q): the value below which q percent of a falls.
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
