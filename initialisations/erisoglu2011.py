"""
Erisoglu 2011 "new" algorithm:

See: A new algorithm for initial cluster centers in k-means algorithm
https://www.sciencedirect.com/science/article/pii/S0167865511002248
"""

from collections import namedtuple

import numpy as np
from scipy.spatial import distance as spdistance

from initialisations.base import Initialisation
import kmeans


class Erisoglu(Initialisation):
    """Erisoglu 2001 initialisation algorithm"""

    def find_centers(self):
        """vi) Turn the candidates into means of initial clusters"""

        first, axes = self._initialise()

        candidates = self._generate_candidates(first, axes)

        distances = kmeans.distance_table(self._data, candidates, axes)
        mins = distances.argmin(1)

        means = [None] * self._num_clusters

        for k in range(self._num_clusters):
            cluster = self._data[mins == k, :]
            means[k] = np.mean(cluster, 0)

        return np.array(means)


    def _find_main_axis(self, data_t):
        """i) Find feature with greatest variance"""

        allvcs = [self.variation_coefficient(feature) for feature in data_t]

        return np.argmax(allvcs)


    def _find_secondary_axis(self, data_t, main_axis):
        """ii) Find feature with least absolute correlation to the main axis"""

        allccs = [abs(self.correlation_coefficient(data_t[main_axis], feature))
                  for feature in data_t]

        return np.argmin(allccs)

    @staticmethod
    def _find_center(data_t, axes):
        """iii) Find the centre point of the data"""

        return [np.mean(data_t[axes.main]), np.mean(data_t[axes.secondary])]


    def _initialise(self):
        """iv) Find data point most remote from center"""

        main = self._find_main_axis(self._data.T)
        secondary = self._find_secondary_axis(self._data.T, main)

        Axes = namedtuple('Axes', 'main secondary')
        axes = Axes(main, secondary)

        center = self._find_center(self._data.T, axes)
        first = self._find_most_remote_from_center(self._data, center, axes)

        return first, axes


    def _generate_candidates(self, first, axes):
        """v) Incrementally find most remote points from latest seed"""

        seeds = [first]

        while len(seeds) < self._num_clusters:
            nextseed = self._find_most_remote_from_seeds(self._data,
                                                         seeds,
                                                         axes)
            seeds.append(nextseed)

        return self._data[seeds]


    def _find_most_remote_from_seeds(self, data, seeds, axes):

        strippedseeds = [[data[seed][axes.main], data[seed][axes.secondary]]
                         for seed in seeds]

        alldists = [self.distance(np.array([entity[axes.main],
                                            entity[axes.secondary]]),
                                  *strippedseeds)
                    for entity in data]

        return np.argmax(alldists)


    def _find_most_remote_from_center(self, data, center, axes):

        alldists = [self.distance(center,
                                  [entity[axes.main], entity[axes.secondary]])
                    for entity in data]

        return np.argmax(alldists)

    # Supporting calculations etc ---------------------------------------------

    @staticmethod
    def variation_coefficient(vector):
        """Absolute value of std dev / mean."""

        return abs(np.std(vector) / np.mean(vector))

    @staticmethod
    def correlation_coefficient(left, right):
        """Correlation coefficient between two vectors"""

        # nb. interesting vectorised implementation:
        # https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/

        numerator = denominator_left = denominator_right = 0

        for val_l, val_r in zip(left, right):

            dev_left = (val_l - np.mean(left))
            dev_right = (val_r - np.mean(right))

            numerator += dev_left * dev_right

            denominator_left += dev_left ** 2
            denominator_right += dev_right ** 2

        # NB: This is where Erisoglu seems to differ from Pearson
        denominator = denominator_left**0.5 * denominator_right**0.5

        return numerator / denominator

    @staticmethod
    def distance(left, *right):
        """Sum of Euclidean distances between a given point and n others"""

        return sum([spdistance.euclidean(left, point) for point in right])


# -----------------------------------------------------------------------------


def generate(data, num_clusters, opts):
    """The common interface"""

    alg = Erisoglu(data, num_clusters, opts)
    return alg.find_centers()
