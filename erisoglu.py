import numpy as np
#import scipy as sp

# Erisoglu 2011 "new" algorithm:
# See: A new algorithm for initial cluster centers in k-means algorithm
# https://www.sciencedirect.com/science/article/pii/S0167865511002248

class Erisoglu():

    # Main steps of the algorithm
    # nb. these assume the data to be an ndarray and transposed

    def find_main_axis(self, data):
        '''Step i) Find the feature with greatest variance'''

        max = 0
        max_column = None

        for j in range(0, len(data)):

            cvj = self.variation_coefficient(data[j])

            if cvj > max:
                max = cvj
                max_column = j

        return max_column


    # Supporting calculations etc ----------------------------------------------

    def variation_coefficient(self, vector):
        '''Absolute value of std dev / mean.'''

        return abs(np.std(vector) / np.mean(vector))


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


    def _prepare_data(self, data):
        '''Since we're working with columns, it's simpler if we transpose first'''

        return np.array(data).T

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    pass
