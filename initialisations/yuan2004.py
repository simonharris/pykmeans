"""
Yuan et al 2004 "new" algorithm

See: A new algorithm to get the initial centroids
https://ieeexplore.ieee.org/abstract/document/1382371
"""

import numpy as np
from scipy.spatial import distance as spdistance

ALPHA = 0.75

# This seems a little brute force and may not scale. Maybe try to find a
# better (eg. vectorised?) solution when time permits


def distance_table(data):
    """Calculate distances between each data point"""

    numrows = len(data)

    distances = np.nan * np.empty((numrows, numrows))

    for i in range(0, numrows):
        for j in range(0, numrows):
            if i != j:
                distances[i][j] = spdistance.euclidean(data[i], data[j])

    return distances


def find_closest(data):
    """Find the closest two data points in the dataset"""

    distances = distance_table(data)
    ind = np.unravel_index(np.nanargmin(distances, axis=None), distances.shape)

    return list(ind)


def find_next_closest(mydata, pointset):
    """Find the point nearest to an already discovered subset"""

    mean = np.mean(mydata, 0)
    return np.argmin([spdistance.euclidean(mean, point) for point in pointset])


def generate(data, num_clusters):
    """The common interface"""

    # Holder for the point sets, called A in the paper
    pointsets = []

    mydata = data.copy()  # U in the paper
    num_points = len(mydata)

    # for each cluster
    while len(pointsets) < num_clusters:

        pointset = []  # Am in the paper
        pair = find_closest(mydata)

        pointset.append(mydata[pair[0]])
        pointset.append(mydata[pair[1]])

        mydata = np.delete(mydata, list(pair), 0)

        while len(pointset) < (ALPHA * (num_points/num_clusters)):
            next_closest = find_next_closest(mydata, pointset)
            pointset.append(mydata[next_closest])
            mydata = np.delete(mydata, next_closest, 0)

        pointsets.append(pointset)

    return np.mean(pointsets, 1)
