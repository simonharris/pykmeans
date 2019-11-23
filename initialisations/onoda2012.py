"""
Onoda 2012 ICA- and PCA-based algorithm

See: Careful seeding method based on independent components analysis for
 k-means clustering
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.663.5343&rep=rep1&type=pdf#page=53
"""

from abc import abstractmethod

import numpy as np

from initialisations.base import Initialisation


class Onoda(Initialisation):
    """Base class for the two Onoda 2012 initialisation algorithms"""

    def _find_centroids(self, data, components):
        """Step 1b from the algorithms"""

        centroids = []

        for component in components:
            distances = [self._calc_distance(x, component) for x in data]
            centroids.append(data[np.argmin(distances)])

        return np.array(centroids)

    @staticmethod
    def _calc_distance(row, component):
        """Used in Step 1b from the algorithms"""

        mag = np.linalg.norm

        return np.dot(component, row) / (mag(component) * mag(row))

    @abstractmethod
    @staticmethod
    def _find_components(data: np.array, num_clusters: int) -> np.array:
        """Each algorithm must implement this"""

    def find_centers(self, data, num_clusters):
        """Main method"""

        components = self._find_components(data, num_clusters)
        return self._find_centroids(data, components)
