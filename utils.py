import numpy as np
from sklearn import preprocessing

'''
General purpose utils for pycluster
'''

# Data preprocessing functions -------------------------------------------------

def standardise(data):
    '''Scale data from -1 to 1, with 0 mean and unit variance'''

    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    return min_max_scaler.fit_transform(data)
