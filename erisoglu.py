import numpy as np
import utils as pu

#
# Erisoglu 2011 "new" algorithm:
# See: A new algorithm for initial cluster centers in k-means algorithm
# https://www.sciencedirect.com/science/article/pii/S0167865511002248
#

def coefficient_of_variation(vector):
    '''Step i): absolute value of std dev / mean. Data must not be standardised'''
 
    return abs(np.std(vector) / np.mean(vector))
   

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    pass

