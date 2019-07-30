'''Provides a uniform interface to load all datasets'''

import os

import numpy as np


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


def load_wbco():
    return _load_local('wbco')
    
# TODO: Iris

# TODO: Wine

