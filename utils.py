import numpy as np
from sklearn import preprocessing

#
# General purpose utils for pycluster
#

# Data loading functions -------------------------------------------------------


def get_learning_data():
    '''Handy learning data from Renato's CE705 Python assignment'''

    # Note: this will not currently work outside of unit test as it is
    # not a SKLearn Bunch, e.g. no 'target' attribute'
    
    return np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='float')


# Data preprocessing functions -------------------------------------------------


def standardise(data):
    '''Scale data from -1 to 1, with 0 mean and unit variance'''

    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    return min_max_scaler.fit_transform(data)
