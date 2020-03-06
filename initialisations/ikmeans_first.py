"""
Implementation of Intelligent k-means which returns the first K centroids
found, ie. the most anomalous ones.
"""

from initialisations.ikmeans_base import Ikmeans


class IkmeansFirst(Ikmeans):
    """Comment string"""

    def _select_centroids(self, centroids, cardinalities):
        """Select the clusters with highest cardinality"""

        return centroids[0:self._num_clusters]


def generate(data, num_clusters):
    """The common interface"""

    ikm = IkmeansFirst(data, num_clusters)
    return ikm.find_centers()
