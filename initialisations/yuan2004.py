from scipy.spatial import distance as spdistance
import numpy as np

'''
Yuan et al 2004 "new" algorithm

See: A new algorithm to get the initial centroids
https://ieeexplore.ieee.org/abstract/document/1382371
'''

#TODO: this is brute force and won't scale. Try to find better (vectorised?) solution
def distance_table(data):
    '''Calculate distances between each data point'''

    numrows = len(data)

    distances = np.nan * np.empty((numrows, numrows))

    for i in range(0, numrows):
        for j in range(0, numrows):
            if i != j:
                distances[i][j] = spdistance.euclidean(data[i], data[j])

    return distances


def find_closest(data):
    '''Find the closest two data points in the dataset'''

    distances = distance_table(data)
    ind = np.unravel_index(np.nanargmin(distances, axis=None), distances.shape)

    return list(ind)


def find_next_closest(Am, U):
    mean = np.mean(Am, 0)
    return np.argmin([spdistance.euclidean(mean, point) for point in U])


def generate(data, K):
    # Holder for the point sets
    A = []

    U = data.copy()
    n = len(U)
    alpha = 0.75

    while len(A) < K:

        Am = []
        pair = find_closest(U)

        Am.append(U[pair[0]])
        Am.append(U[pair[1]])

        U = np.delete(U, list(pair), 0)

        while len(Am) < (alpha * (n/K)):
            nc = find_next_closest(U, Am)
            Am.append(U[nc])
            np.delete(U, nc)

        A.append(Am)

    return np.mean(A, 1)
