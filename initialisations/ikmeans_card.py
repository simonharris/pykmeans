"""
Implementation of Intelligent k-means which returns the centroids of clusters
with the highest cardinality
"""

import numpy as np

from initialisations.ikmeans_base import Ikmeans


class IkmeansCard(Ikmeans):
    """Comment string"""

    def _select_centroids(self, centroids, cardinalities):
        """Select the clusters with highest cardinality"""

        highest = np.argpartition(cardinalities,
                                  -self._num_clusters)[-self._num_clusters:]

        return np.array([centroids[i] for i in highest])


def generate(data, num_clusters):
    """The common interface"""

    ikm = IkmeansCard(data, num_clusters)
    return ikm.find_centers()
