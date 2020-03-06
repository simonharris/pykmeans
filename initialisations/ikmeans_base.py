"""
Intelligent K-Means clustering algorithm

See: Mirkin 2005 - Clustering for data mining: a data recovery approach

Notes:
 - we skip step 1 (pre-processing) of AP as our data is standardised in
   advance, consistently for all sets and algorithms
 - we've treated the threshold as implicitly 0. Mention this in the paper
"""

from abc import abstractmethod

import numpy as np
import kmeans

from initialisations.base import Initialisation, InitialisationException

# Makes for unreadable code and causes errors
# pylint: disable=C1801


class Ikmeans(Initialisation):
    """Base class for Intelligent k-means implementations"""

    @abstractmethod
    def _select_centroids(self, centroids: np.array,
                          cardinalities: list) -> np.array:
        """Each algorithm must implement this"""

    @staticmethod
    def _find_origin(data):
        """Find the center of the data"""

        return np.mean(data, axis=0)

    def _find_most_distant(self, data):
        """Find the point most distant from the origin"""

        origin = np.array([self._find_origin(data)])
        distances = kmeans.distance_table(data, origin)
        return data[distances.argmax()], origin

    def _anomalous_pattern(self, data):
        """Locate the most anomalous cluster in a data set"""

        # Special case to discuss. Otherwise we get an infinite loop below
        if len(data) == 1:
            return data[0], [0]

        # ii) Initial setting
        # The furthest is called "c" in the paper
        center_c, origin = self._find_most_distant(data)

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

    def find_centers(self):
        """The common interface"""

        # We're about to manipulate this
        data_working = np.copy(self._data)
        centroids = []
        cardinalities = []

        # Stop condition 1
        while len(data_working) > 0:

            centre, assigned_indexes = self._anomalous_pattern(data_working)
            centroids.append(centre)
            cardinalities.append(len(assigned_indexes))

            data_working = np.delete(data_working, assigned_indexes, 0)

        if len(centroids) < self._num_clusters:
            raise InitialisationException(
                "Found only %d/%d centroids" % (len(centroids),
                                                self._num_clusters))

        # delegate to child classes
        return self._select_centroids(centroids, cardinalities)
