"""P3: the call-list metric does what we claim.

Precision@k asks: of the k prospects ranked highest, what share are real positives? These two tests check
the two extremes on a tiny 5-prospect list with 2 positives.
"""
from wtl.metrics import precision_at_k

#
# A perfect ranking puts both positives in the top 2.
#
def test_perfect_ranking_puts_positives_on_top():
    """If the two positives get the two highest scores, precision@2 must be 1.0."""
    y = [1,0,1,0,0]
    scores = [0.9, 0.1, 0.8, 0.3, 0.2] # the two positives have the two highest scores

    assert precision_at_k(y,scores, k = 2) == 1.0

#
# The worst ranking puts both positives at the bottom.
# 

def test_worst_ranking_puts_positives_last():
    """If the two positives get the two lowest scores, precision@2 must be 0.0."""
    y = [1,0,1,0,0]
    scores = [0.1, 0.9, 0.2, 0.8, 0.7]   # the two positives have the two lowest scores

    assert precision_at_k(y, scores, k=2) == 0.0