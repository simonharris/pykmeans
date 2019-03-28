import numpy as np

# Erisoglu 2011 "new" algorithm:
# See: A new algorithm for initial cluster centers in k-means algorithm
# https://www.sciencedirect.com/science/article/pii/S0167865511002248

class Erisoglu():

    # Main steps of the algorithm
    # nb. these assume the data to be an ndarray and transposed

    def find_main_axis(self, data):
        '''i) Find feature with greatest variance'''

        allvcs = np.array([self.variation_coefficient(feature) for feature in data])

        return allvcs.argmax()


    def find_secondary_axis(self, data, main_axis):
        '''ii) Find feature with least absolute correlation to the main axis'''

        main = data[main_axis]
        allccs = np.array([abs(self.correlation_coefficient(main, feature)) for feature in data])

        return allccs.argmin()


    def find_center(self, data):
        '''iii) Find the center point of the data'''

        main, secondary = self._find_both_axes(data)

        return np.mean(data[main]), np.mean(data[secondary])


    def _find_both_axes(self, data):

        main = self.find_main_axis(data)
        secondary = self.find_secondary_axis(data, main)

        return main, secondary

    # Supporting calculations etc ----------------------------------------------

    # No doubt all of these are provided by libraries, but the exercise here is
    # to understand what's happening

    def variation_coefficient(self, vector):
        '''Absolute value of std dev / mean.'''

        return abs(np.std(vector) / np.mean(vector))


    # Interesting vectorised implementation:
    # https://waterprogramming.wordpress.com/2014/06/13/numpy-vectorized-correlation-coefficient/

    def correlation_coefficient(self, left, right): # column j, column j'
        '''Correlation coefficient between two vectors'''

        numerator = 0
        denominator_left = 0
        denominator_right = 0

        for i in range(0, len(left)):

            dev_left = (left[i] - np.mean(left))
            dev_right = (right[i] - np.mean(right))

            numerator +=  dev_left * dev_right

            denominator_left += dev_left ** 2
            denominator_right += dev_right ** 2

        # TODO: This is where Erisoglu seems to differ from Pearson
        denominator = denominator_left**0.5 * denominator_right**0.5
        #denominator = denominator_left * denominator_right

        return (numerator / denominator)


    def euclidean_distance(self, left, right):
        '''Already implemented this for CE705, so won't spend time here :)'''

        return np.linalg.norm(left - right, axis=0)


    # basic utilities ----------------------------------------------------------

    def _prepare_data(self, data):
        '''Since we're working with columns, it's simpler if we transpose first'''

        return np.array(data).T
