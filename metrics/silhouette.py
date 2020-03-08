"""Adjusted Rand Index score"""

import sklearn.metrics as skmetrics


def score(target, found):
    """The common interface"""

    return skmetrics.silhouette_score(target, found)
