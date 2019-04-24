import numpy as np

#TODO: ask Renato the name of the algorithm

def from_matrix(mat):
    return np.sum(np.max(mat, 0)) / np.sum(mat)
