"""
Intelligent K-Means clustering algorithm

See: Mirkin 2005 - Clustering for data mining: a data recovery approach
"""

import numpy as np
from sklearn import preprocessing
import kmeans


def _find_origin(data):
    """Find the center of the data (will be 0 if standardised)"""

    return np.mean(data, axis=0)


def _find_most_distant(data):
    """Find the point most distant from the origin"""

    origin = np.array([_find_origin(data)])
    distances = kmeans.distance_table(data, origin)
    return data[distances.argmax()], origin


def _anomalous_pattern(data):
    """Locate the most anomalous cluster in a data set"""

    # ii) Initial setting
    # furthest is called "c" in the paper
    center_c, origin = _find_most_distant(data)

    # iii) Cluster update
    while True:

        all_dist = kmeans.distance_table(data,
                                         np.array([origin[0], center_c]))
        partition = all_dist.argmin(1)

        # Needed later to remove from data. Callued "Ui"
        assigned_indexes = np.where(partition == 1)[0]

        # Called "S" in the paper
        cluster_s = data[partition == 1, :]

        # iv) Centroid update
        center_tentative = np.mean(cluster_s, 0)

        if np.array_equal(center_c, center_tentative):
            break
        else:
            center_c = center_tentative

    # v) Output
    return center_c, assigned_indexes


def generate(data, num_clusters):
    """The common interface"""

    # i) Standardise the original data only at t=1
    # Datasets are already standardised

    data_working = data
    centroids = []

    while True:

        centre, assigned_indexes = _anomalous_pattern(data_working)

        centroids.append(centre)

        data_working = np.delete(data_working, assigned_indexes, 0)

        # Stopping condition #4
        if len(centroids) >= num_clusters:
            break

    return np.array(centroids)
