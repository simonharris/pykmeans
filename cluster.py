'''General purpose cluster class'''

import numpy as np
from scipy.spatial import distance as spdistance


class Cluster():
    """Models a cluster and its assigned vectors"""

    def __init__(self):
        self._mean = None
        self._samples = []

    def assign(self, vector, recalc_mean=True):
        """Assign a vector to the cluster"""

        self._samples.append(vector)

        if recalc_mean:
            self._calculate_mean()

    def get_samples(self):
        """Return all vectors in the cluster"""

        return self._samples

    def get_mean(self):
        """Mean/centroid of the cluster"""

        return self._mean

    def get_distance(self, vector):
        """Distance between the centroid and a supplied vector"""

        return spdistance.euclidean(self.get_mean(), vector)

    def merge(self, other):
        """Merge another cluster into this one"""

        for sample in other.get_samples():
            self.assign(sample, recalc_mean=False)

        # just done once at the end for efficiency
        self._calculate_mean()

    def _calculate_mean(self):
        self._mean = np.mean(np.array(self._samples), axis=0)

    def __str__(self):
        return str(self._mean)
