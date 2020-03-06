"""
Yuan et al 2004 "new" algorithm

See: A new algorithm to get the initial centroids
https://ieeexplore.ieee.org/abstract/document/1382371
"""

import numpy as np
from scipy.spatial import distance as spdistance

ALPHA = 0.75


def distance_table(data):
    """Calculate distances between each data point"""

    numrows = len(data)

    distances = np.nan * np.empty((numrows, numrows))

    for point in range(numrows):
        distances[:, point] = np.sum((data - data[point, :])**2, 1)**0.5

    # feels a little hacky, but getting "min where not zero" is even uglier
    distances[distances == 0] = np.nan

    return distances


def find_closest(data):
    """Find the closest two data points in the dataset"""

    distances = distance_table(data)
    ind = np.unravel_index(np.nanargmin(distances, axis=None), distances.shape)

    return list(ind)


def find_next_closest(latestdata, pointset):
    """Find the point nearest to an already discovered subset"""

    psmean = np.mean(pointset, 0)
    distances = [spdistance.euclidean(psmean, point) for point in latestdata]

    return np.argmin(distances)


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

        mydata = np.delete(mydata, pair, axis=0)

        desired_points = ALPHA * (num_points/num_clusters)

        while len(pointset) < desired_points:

            next_closest = find_next_closest(mydata, pointset)

            pointset.append(mydata[next_closest])
            mydata = np.delete(mydata, next_closest, axis=0)

        pointsets.append(pointset)

    return np.mean(pointsets, 1)
