import numpy as np
from scipy.spatial import distance as spdistance

# Erisoglu 2011 "new" algorithm:
# See: A new algorithm for initial cluster centers in k-means algorithm
# https://www.sciencedirect.com/science/article/pii/S0167865511002248

class Erisoglu():

    def find_main_axis(self, data):
        '''i) Find feature with greatest variance'''

        allvcs = [self.variation_coefficient(feature) for feature in data]

        return np.argmax(allvcs)


    def find_secondary_axis(self, data, main_axis):
        '''ii) Find feature with least absolute correlation to the main axis'''

        main = data[main_axis]
        allccs = [abs(self.correlation_coefficient(main, feature)) for feature in data]

        return np.argmin(allccs)


    def find_center(self, data, main, secondary):
        '''iii) Find the center point of the data'''

        return [np.mean(data[main]), np.mean(data[secondary])]


    def find_initial_seed(self, data):
        '''iv) Find data point most remote from center'''

        self._main = self.find_main_axis(data.T)
        self._secondary = self.find_secondary_axis(data.T, self._main)
        center = self.find_center(data.T, self._main, self._secondary)

        return self._find_most_remote_by_value(data, center, self._main, self._secondary)


    def generate(self, data, K):
        '''v) Incrementally find most remote points from latest seed'''

        seeds = []
        seed = self.find_initial_seed(data)
        seeds.append(seed)

        while (len(seeds) < K):
            nextseed = self._find_most_remote_row(data, seeds)

            seeds.append(nextseed)

        return data[seeds]


    def _find_most_remote_row(self, data, seeds):

        main, secondary = self._main, self._secondary

        strippedseeds = np.array([ [data[seed][main], data[seed][secondary]] for seed in seeds ])

        alldists = [self.distance(np.array([entity[main], entity[secondary]]), *strippedseeds)
                            for entity in data]

        return np.argmax(alldists)


    def _find_most_remote_by_value(self, data, start, main, secondary):

        alldists = [self.distance(start, [entity[main], entity[secondary]])
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

        numerator = 0
        denominator_left = 0
        denominator_right = 0

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
        '''Sum of Euclidean distances between a given point and n others'''

        dist = 0

        for point in right:
            dist += spdistance.euclidean(left, point)

        return dist
