"""
Khan 2012 initial seed selection algorithm

See:
An initial seed selection algorithm for k-means clustering of georeferenced
data to improve replicability of cluster assignments for mapping application
https://www.sciencedirect.com/science/article/pii/S1568494612003377
"""

import numpy as np
from scipy.spatial import distance as spdistance


def sort_by_magnitude(data, column):
    """ Step a) Sort data by magnitude of given column"""
    # idxs = np.argsort(np.linalg.norm(data, axis=1))
    # return data[idxs]

    return np.array(sorted(data, key=lambda row: np.abs(row[column])))


def find_distances(data, column):
    """Step b) Find Euclidean distances between sorted rows"""

    datacol = data[:, column]

    distances = []

    for i in range(0, len(data)-1):
        distances.append(spdistance.euclidean(datacol[i], datacol[i+1]))

    return np.array(distances)


def find_split_points(distances, num_clusters):
    """Step c) Find the largest distances between adjacent rows"""

    howmany = num_clusters - 1

    return distances.argsort()[-howmany:][::-1]


def generate(data, num_clusters, column):
    """The common interface"""

    # Step a)
    sorteddata = sort_by_magnitude(data, column)

    # Step b)
    distances = find_distances(sorteddata, column)

    # Step c)
    splits = find_split_points(distances, num_clusters)

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
