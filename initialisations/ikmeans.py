"""
Intelligent K-Means clustering algorithm

See: Mirkin 2005 - Clustering for data mining: a data recovery approach

Notes:
 - we skip step 1 (pre-processing) of AP as our data is standardised in
   advance, consistently for all sets and algorithms
"""

import numpy as np
import kmeans

from initialisations.base import InitialisationException

# TODO:
#   ??- put threshold=0 (as per Renato) and add a note to LaTex
#   - - and do we somehow check this?
#   DONE - change stopping condition to "run out of data"
#   - some kind of check whether we even reach K, esp. when it's 20
#   DONE - *then* go into selecting which ones. Renato gave two choices


def _find_origin(data):
    """Find the center of the data"""

    return np.mean(data, axis=0)


def _find_most_distant(data):
    """Find the point most distant from the origin"""

    origin = np.array([_find_origin(data)])
    distances = kmeans.distance_table(data, origin)
    return data[distances.argmax()], origin


def _anomalous_pattern(data):
    """Locate the most anomalous cluster in a data set"""

    # Special case to discuss. Otherwise we get an infinite loop below
    if len(data) == 1:
        return data[0], [0]

    # ii) Initial setting
    # The furthest is called "c" in the paper
    center_c, origin = _find_most_distant(data)

    # iii) Cluster update
    while True:

        all_dist = kmeans.distance_table(data,
                                         np.array([origin[0], center_c]))
        partition = all_dist.argmin(1)

        # Needed later to remove from data. Called "Ui"
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

    data_working = data
    centroids = []
    cardinalities = []

    # Stop condition 1
    while len(data_working) > 0:

        centre, assigned_indexes = _anomalous_pattern(data_working)
        centroids.append(centre)
        cardinalities.append(len(assigned_indexes))

        data_working = np.delete(data_working, assigned_indexes, 0)

        # This was one of the two options for selecting which ones to use
        # Stopping condition #4
        '''if len(centroids) >= num_clusters:
            print("Got enough while data len is:", len(data_working))
            break'''

    # print(cardinalities)

    if len(centroids) < num_clusters:
        raise InitialisationException(
            "Found only %d/%d centroids" % (len(centroids), num_clusters))

    # This was the other centroid selection option...
    highest = np.argpartition(cardinalities, -num_clusters)[-num_clusters:]

    # print(highest)

    final_centroids = [centroids[i] for i in highest]

    return np.array(final_centroids)
