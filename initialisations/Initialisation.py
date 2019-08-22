"""
Base class to encourage consistency of structure between Initialisations
"""

from abc import abstractmethod

import numpy as np

class Initialisation:


    def __init__(self, data: np.array, K: int, opts: dict):
        self._data = data
        self._K = K
        self._opts = opts
        self._num_samples = data.shape[0]
        self._num_attrs = data.shape[1]
        

    @abstractmethod
    def find_centers(self) -> np.array:
        pass       

