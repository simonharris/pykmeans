'''Adjusted Rand Index score'''

import sklearn.metrics as skmetrics


def score(target, found):
    return skmetrics.adjusted_rand_score(target, found)
    
