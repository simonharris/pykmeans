"""
Base class to encourage consistency of structure between Initialisations
"""

from abc import abstractmethod

import numpy as np


class Initialisation:
    """Common structure for all initialisations"""

    def __init__(self, data: np.array, num_clusters: int):
        self._data = data
        self._K = num_clusters  # DEPRECATED: use num_clusters
        self._num_clusters = num_clusters
        self._opts = {}
        self._num_samples = data.shape[0]
        self._num_attrs = data.shape[1]

    @abstractmethod
    def find_centers(self, data: np.array, num_clusters: int) -> np.array:
        """The main method that all initialisations must implement"""
