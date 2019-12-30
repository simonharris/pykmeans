"""
Khan 2012 initial seed selection algorithm

See:
An initial seed selection algorithm for k-means clustering of georeferenced
data to improve replicability of cluster assignments for mapping application
https://www.sciencedirect.com/science/article/pii/S1568494612003377
"""

import numpy as np
from scipy.spatial import distance as spdistance

# This is potentially problematic. The choice of column seems arbitrary, and
# is specific to dataset, which whill not scale to 6,030 datasets
COLUMN = 0


def _sort_by_magnitude(data, column):
    """ Step a) Sort data by magnitude of given column"""
    # idxs = np.argsort(np.linalg.norm(data, axis=1))
    # return data[idxs]

    return np.array(sorted(data, key=lambda row: np.abs(row[column])))


def _find_distances(data, column):
    """Step b) Find Euclidean distances between sorted rows"""

    datacol = data[:, column]

    distances = []

    for i in range(0, len(data)-1):
        distances.append(spdistance.euclidean(datacol[i], datacol[i+1]))

    return np.array(distances)


def _find_split_points(distances, num_clusters):
    """Step c) Find the largest distances between adjacent rows"""

    howmany = num_clusters - 1

    return distances.argsort()[-howmany:][::-1]


def generate(data, num_clusters):
    """The common interface"""

    # Step a)
    sorteddata = _sort_by_magnitude(data, COLUMN)

    # Step b)
    distances = _find_distances(sorteddata, COLUMN)

    # Step c)
    splits = _find_split_points(distances, num_clusters)

    # Step d) Find upper bounds

    # Because there n-1 possible splits
    uppers = [split + 1 for split in splits]
    # and the final upper is always n
    uppers = np.append(uppers, len(sorteddata))
    uppers.sort()

    # Step e and f)
    centroids = []

    lower = 0

    for i in range(0, num_clusters):
        upper = uppers[i]
        group = sorteddata[lower:upper]
        centroids.append(np.mean(group, axis=0))
        lower = upper

    return np.array(centroids)
