import numpy as np
from scipy.spatial import distance as spdistance
from collections import namedtuple

# Erisoglu 2011 "new" algorithm:
# See: A new algorithm for initial cluster centers in k-means algorithm
# https://www.sciencedirect.com/science/article/pii/S0167865511002248

class Erisoglu():

    def _initialise(self, data):

        # i) Find feature with greatest variance
        main = np.argmax([self.variation_coefficient(feature) for feature in data.T])

        # ii) Find feature with least absolute correlation to the main axis
        secondary = np.argmin([abs(self.correlation_coefficient(data.T[main], feature))
                                for feature in data.T])

        axes = namedtuple('Axes', 'main secondary')(main, secondary)

        # iii) Find the center point of the data'''
        center = [np.mean(data.T[axes.main]), np.mean(data.T[axes.secondary])]

        # iv) Find data point most remote from center
        first = self._find_most_remote_from_center(data, center, axes)

        return first, axes


    def generate(self, data, K):

        # v) Incrementally find most remote points from latest seed

        first, axes = self._initialise(data)
        seeds = [first]

        while (len(seeds) < K):
            nextseed = self._find_most_remote_from_seeds(data, seeds, axes)
            seeds.append(nextseed)

        return data[seeds]


    def _find_most_remote_from_center(self, data, center, axes):
        alldists = [self.distance(center, [entity[axes.main], entity[axes.secondary]])
                 for entity in data]

        return np.argmax(alldists)


    def _find_most_remote_from_seeds(self, data, seeds, axes):
        strippedseeds = [[data[seed][axes.main], data[seed][axes.secondary]] for seed in seeds]
        alldists = [self.distance(np.array([entity[axes.main], entity[axes.secondary]]), *strippedseeds)
                            for entity in data]

        return np.argmax(alldists)

    # Supporting calculations etc ----------------------------------------------

    def variation_coefficient(self, vector):
        '''Absolute value of std dev / mean.'''

        return abs(np.std(vector) / np.mean(vector))


    def correlation_coefficient(self, left, right):
        '''Correlation coefficient between two vectors'''

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

        return (numerator / denominator)


    def distance(self, left, *right):
        '''Sum of Euclidean distances between a given point and n others'''

        return sum([spdistance.euclidean(left, point) for point in right])

# For more consistent integration with the notebook ----------------------------

def generate(data, K):
    e = Erisoglu()
    return e.generate(data, K)
