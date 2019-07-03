import numpy as np


def generate(data, K):
    '''Select random data points as initial seeds'''

    return data[np.random.choice(data.shape[0], K, replace=False), :]
