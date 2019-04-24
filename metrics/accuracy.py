import numpy as np

#TODO: ask Renato the name of the algorithm

def from_matrix(mat):

    total = np.sum(mat)
    hits = 0

    while len(mat) > 0:

        row, col = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        hits += mat[row, col]

        mat = np.delete(mat, row, axis=0)
        mat = np.delete(mat, col, axis=1)

    return hits/total
