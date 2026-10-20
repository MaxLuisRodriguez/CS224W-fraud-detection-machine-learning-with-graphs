"""Ranking metrics for call lists.

A sales rep can only make the first K calls of a list, so these metrics score how good the TOP of a 
ranking is, on lists the size of a real call list, not just the whole dataset at once.
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
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    
    # sorted in ascending order, so need to take negative of scores to get highest, or just grab length-k and beyond
    top = np.argsort(-scores, kind="stable")[:k] # indices of the k highest scores (stable: ties keep input order)

    return float(y_true[top].mean())


#
# Call-list metrics: AUROC, average precision, and precision@K, averaged over ranodm call lists
#
def call_list_metrics(y_true, scores, list_size: int = 128, num_lists: int = 200, seed: int = 0) -> dict:
    """Score a model the way a rep experiences it: on many random call lists of `list_size` prospects.

        For each of `num_lists` random lists (drawn without replacement from all scored prospects):
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
        idx = rng.choice(len(y_true), size=list_size, replace=False) # one random call list
        y, s = y_true[idx], scores[idx]

        p_at_k.append(precision_at_k(y,s,k)) # always defined

        if 0 < y.sum() < len(y):
            auroc.append(roc_auc_score(y, s))
            ap.append(average_precision_score(y, s))

    return {"auroc": float(np.mean(auroc)), "ap": float(np.mean(ap)), f"p@{k}": float(np.mean(p_at_k)),
            "lists_scored": len(auroc)}