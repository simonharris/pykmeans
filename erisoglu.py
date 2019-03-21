import numpy as np
import utils as pu

#
# Erisoglu 2011: A new algorithm for initial cluster centers in k-means algorithm
# See: https://www.sciencedirect.com/science/article/pii/S0167865511002248
#

def variation_coefficient(vector):
    '''Description of function and note about please, no standardised data'''

    # stddev
    S = np.std(vector)
    print("stddev is:", S, "\n")

    # mean = ...

    # foo = stddev / mean

    # bar = abs(foo)

    # return bar




# ------------------------------------------------------------------------------

if __name__ == '__main__':


    data = [1, 2, 3]

    foo = variation_coefficient(data)
    print("CVj is:", foo)

    #data = pu.get_learning_data()

