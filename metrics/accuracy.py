import numpy as np
import sklearn.metrics as skmetrics

def from_matrix(mat):
    return np.sum(np.max(mat, 0)) / np.sum(mat)

def score(target, found):
    return from_matrix(skmetrics.confusion_matrix(target, found))
