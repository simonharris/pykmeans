"""
Intelligent K-Means clustering algorithm

See: Mirkin 2005 - Clustering for data mining: a data recovery approach
"""

import numpy as np
from sklearn import preprocessing
import kmeans


def anomalous_pattern(data):
    """Locate the most anomalous cluster in a data set"""

    # i) Standardise the original data (idempotent)
    data = standardise(data)

    # ii) Initial setting
    origin = np.zeros((1, data.shape[1]))

    initial_dist = kmeans.distance_table(data, origin)

    furthest = data[initial_dist.argmax()]  # called "c" in the paper

    # iii) Cluster update
    while True:

        all_dist = kmeans.distance_table(data,
                                         np.array([origin[0], furthest]))
        partition = all_dist.argmin(1)

        # Needed later to remove from data. Callued "Ui"
        partition_i = np.where(partition == 1)

        # Called "S" in the paper
        cluster_list = data[partition == 1, :]

        # iv) Centroid update
        c_tentative = np.mean(cluster_list, 0)

        if np.array_equal(furthest, c_tentative):
            break
        else:
            # TODO: does the name still make sense?
            furthest = c_tentative

    # v) Output
    return furthest, partition_i


def generate(data, num_clusters):
    """The common interface"""

    data_working = data
    centroids = []

    while True:

        centre, partition_i = anomalous_pattern(data_working)

        centroids.append(centre)

        data_working = np.delete(data_working, partition_i, 0)

        # TODO: investigate other stopping conditions
        if len(centroids) >= num_clusters:
            break

    return np.array(centroids)


def standardise(data):
    """Scale data from -1 to 1, with 0 mean and unit variance"""

    min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
    return min_max_scaler.fit_transform(data)
