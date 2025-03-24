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

    def _find_centroids(self, components) -> np.array:
        """Step 1b from the algorithms"""

        centroids = []

        for component in components:
            distances = [self._calc_distance(x, component) for x in self._data]
            centroids.append(self._data[np.argmin(distances)])

        return np.array(centroids)

    @staticmethod
    def _calc_distance(row, component):
        """Used in Step 1b from the algorithms"""

        mag = np.linalg.norm

        return np.dot(component, row) / (mag(component) * mag(row))

    @staticmethod
    @abstractmethod
    def _find_components() -> np.array:
        """Each algorithm must implement this"""

    def find_centers(self):
        """Main method"""

        return self._find_centroids(self._find_components())
