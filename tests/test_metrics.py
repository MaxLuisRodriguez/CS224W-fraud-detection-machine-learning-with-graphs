"""P3: the call-list metric does what we claim."""
from wtl.metrics import precision_at_k

#
# A perfect ranking puts both positives in the top 2.
#
def test_perfect_ranking_puts_positives_on_top():
    y = [1,0,1,0,0]
    scores = [0.9, 0.1, 0.8, 0.3, 0.2] # the two positives have the two highest scores

    assert precision_at_k(y,scores, k = 2) == 1.0

#
# The worst ranking puts both positives at the bottom.
# 

def test_worst_ranking_puts_positives_last():
    y = [1,0,1,0,0]
    scores = [0.1, 0.9, 0.2, 0.8, 0.7]   # the thwo positives have the two lowest scores

    assert precision_at_k(y, scores, k=2) == 0.0