"""Attempt at calculating accuracy score"""

import numpy as np
import sklearn.metrics as skmetrics


def from_matrix(mat):
    """Calculate from a confusion matrix"""
    return np.sum(np.max(mat, 0)) / np.sum(mat)


def score(target, found):
    """The common interface"""

    return from_matrix(skmetrics.confusion_matrix(target, found))
