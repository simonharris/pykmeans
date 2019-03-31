import numpy as np

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


    def find_most_remote(self, data):
        '''iv) Find data point most remote from center'''

        main = self.find_main_axis(data.T)
        secondary = self.find_secondary_axis(data.T, main)
        center = self.find_center(data.T, main, secondary)

        alldists = [self.euclidean_distance(center, [feature[main], feature[secondary]])
                 for feature in data]

        return np.argmax(alldists)


    # Supporting calculations etc ----------------------------------------------

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

        # NB: This is where Erisoglu seems to differ from Pearson
        denominator = denominator_left**0.5 * denominator_right**0.5
        #denominator = denominator_left * denominator_right

        return (numerator / denominator)


    def euclidean_distance(self, left, right):
        '''Implemented this for CE705, so let's just use the libraries this time'''

        return np.linalg.norm(np.array(left) - np.array(right), axis=0)
