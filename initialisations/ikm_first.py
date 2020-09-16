"""
Implementation of Intelligent k-means which returns the first K centroids
found, ie. the most anomalous ones.
"""

import numpy as np

from initialisations.ikm_base import Ikmeans


class IkmeansFirst(Ikmeans):
    """Select the most anomalous clusters"""

    def _select_centroids(self, centroids, cardinalities):
        """Select the clusters with highest cardinality"""

        return np.array(centroids[0:self._num_clusters])


def generate(data, num_clusters):
    """The common interface"""

    ikm = IkmeansFirst(data, num_clusters)
    return ikm.find_centers()
