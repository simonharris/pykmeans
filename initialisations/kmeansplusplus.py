"""
Arthur & Vassilvitskii k-means++ algorithm

See: k-means++: The advantages of careful seeding
http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
"""

import numpy as np

from initialisations.base import Initialisation
import kmeans


class Kmeansplusplus(Initialisation):
    """k-means++ initialisation algorithm"""

    def find_centers(self):

        # Initial centroid
        randindex = np.random.choice(self._num_samples, replace=False)
        centroids = np.array([self._data[randindex]])

        # Remaining required centroids
        while len(centroids) < self._num_clusters:

            distances = kmeans.distance_table(self._data, centroids)
            probabilities = distances.min(1)**2 / np.sum(distances.min(1)**2)

            randindex = np.random.choice(self._num_samples,
                                         replace=False,
                                         p=probabilities)
            centroids = np.append(centroids, [self._data[randindex]], 0)

        return centroids


def generate(data, num_clusters):
    """The common interface"""

    kmpp = Kmeansplusplus(data, num_clusters)
    return kmpp.find_centers()
