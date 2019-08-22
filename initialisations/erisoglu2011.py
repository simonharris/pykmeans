import numpy as np
from scipy.spatial import distance as spdistance
from collections import namedtuple
import kmeans

"""
Erisoglu 2011 "new" algorithm:

See: A new algorithm for initial cluster centers in k-means algorithm
https://www.sciencedirect.com/science/article/pii/S0167865511002248
"""

from initialisations.Initialisation import Initialisation


class Erisoglu(Initialisation):


    def find_centers(self):
        """vi) Turn the candidates into means of initial clusters"""

        first, axes = self._initialise()

        candidates = self._generate_candidates(self._data, self._K, first, axes)

        distances = kmeans.distance_table(self._data, candidates, axes)
        mins = distances.argmin(1)

        M = [None] * self._K

        for k in range(self._K):
            cluster = self._data[mins==k, :]
            M[k] = np.mean(cluster, 0)

        return np.array(M)


    def _find_main_axis(self, dataT):
        """i) Find feature with greatest variance"""

        allvcs = [self.variation_coefficient(feature) for feature in dataT]

        return np.argmax(allvcs)


    def _find_secondary_axis(self, dataT, main_axis):
        """ii) Find feature with least absolute correlation to the main axis"""

        allccs = [abs(self.correlation_coefficient(dataT[main_axis], feature)) 
                            for feature in dataT]

        return np.argmin(allccs)


    def _find_center(self, dataT, axes):
        """iii) Find the centre point of the data"""

        return [np.mean(dataT[axes.main]), np.mean(dataT[axes.secondary])]


    def _initialise(self):
        """iv) Find data point most remote from center"""

        main = self._find_main_axis(self._data.T)
        secondary = self._find_secondary_axis(self._data.T, main)

        Axes = namedtuple('Axes', 'main secondary')
        axes = Axes(main, secondary)

        center = self._find_center(self._data.T, axes)
        first = self._find_most_remote_from_center(self._data, center, axes)

        return first, axes


    def _generate_candidates(self, data, K, first, axes):
        """v) Incrementally find most remote points from latest seed"""

        seeds = [first]

        while (len(seeds) < K):
            nextseed = self._find_most_remote_from_seeds(data, seeds, axes)
            seeds.append(nextseed)

        return data[seeds]


    def _find_most_remote_from_seeds(self, data, seeds, axes):

        strippedseeds = [ [data[seed][axes.main], data[seed][axes.secondary]] 
                            for seed in seeds ]

        alldists = [self.distance(np.array([entity[axes.main], entity[axes.secondary]]), *strippedseeds)
                            for entity in data]

        return np.argmax(alldists)


    def _find_most_remote_from_center(self, data, center, axes):

        alldists = [self.distance(center, [entity[axes.main], entity[axes.secondary]])
                 for entity in data]

        return np.argmax(alldists)

    # Supporting calculations etc ----------------------------------------------

    def variation_coefficient(self, vector):
        """Absolute value of std dev / mean."""

        return abs(np.std(vector) / np.mean(vector))


    def correlation_coefficient(self, left, right):
        """Correlation coefficient between two vectors"""

        # nb. interesting vectorised implementation:
        # https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/

        numerator = denominator_left = denominator_right = 0

        for i in range(0, len(left)):

            dev_left = (left[i] - np.mean(left))
            dev_right = (right[i] - np.mean(right))

            numerator +=  dev_left * dev_right

            denominator_left += dev_left ** 2
            denominator_right += dev_right ** 2

        # NB: This is where Erisoglu seems to differ from Pearson
        denominator = denominator_left**0.5 * denominator_right**0.5
        #denominator = denominator_left * denominator_right

        return (numerator / denominator)


    def distance(self, left, *right):
        """Sum of Euclidean distances between a given point and n others"""

        return sum([spdistance.euclidean(left, point) for point in right])


## -----------------------------------------------------------------------------


def generate(data, K, opts):
    """The common interface"""

    e = Erisoglu(data, K, opts)
    return e.find_centers()
    
