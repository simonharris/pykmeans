"""
Base class to encourage consistency of structure between Initialisations
"""

from abc import abstractmethod

import numpy as np
from sklearn.cluster import KMeans


class Initialisation:
    """Common structure for all initialisations"""

    def __init__(self, data: np.array, num_clusters: int):
        self._data = data
        self._num_clusters = num_clusters
        self._num_samples = data.shape[0]
        self._num_attrs = data.shape[1]

    @abstractmethod
    def find_centers(self) -> np.array:
        """The main method that all initialisations must implement"""

    def _run_k_means(self, seeds: np.array) -> np.array:
        """Provide a simplified interface to KMeans"""

        est = KMeans(n_clusters=self._num_clusters, n_init=1, init=seeds)
        est.fit(self._data)

        return est.labels_, est.cluster_centers_, est.inertia_


class EmptyClusterException(Exception):
    """If empty clusters cannot be avoided in current circumstances"""


class InitialisationException(Exception):
    """Various other things went wrong with initialisation"""
