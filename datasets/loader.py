'''Provides a uniform interface to load all datasets'''

import os

import arff
import numpy as np
import pandas as pd
import sklearn.datasets as skdatasets


dir_path = os.path.dirname(os.path.realpath(__file__))


class Dataset():
    '''Mimics the sklearn dataset interface'''

    def __init__(self, data, target):
        self.data = data
        self.target = target


def _load_local(which):

    datafile = dir_path + '/' + which + '/data.csv'
    labelfile = dir_path + '/' + which + '/labels.csv'

    return Dataset(
        np.loadtxt(datafile, delimiter=',', dtype=np.int),
        np.loadtxt(labelfile, delimiter=',', dtype=np.int),
    )


# The loaders to be exposed ---------------------------------------------------

def load_fossil():
    return _load_local('fossil')


def load_hartigan():
    return _load_local('hartigan1975')


def load_iris():
    return skdatasets.load_iris()


def load_iris_ccia():
    """To ensure using the exact same Iris data as Khan & Ahmad 2004"""

    datafile = dir_path + '/iris_ccia/iris.arff'

    with open(datafile) as df:
        iris = np.array(arff.load(df)['data'])

    return Dataset(iris[:, 0:4].astype('float'), pd.factorize(iris[:, 4]))


def load_soy_small():
    return _load_local('soy_small')


def load_wbco():
    return _load_local('wbco')


def load_wine():
    return skdatasets.load_wine()
