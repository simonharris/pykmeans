'''Provides a uniform interface to load all datasets'''

import os

import numpy as np
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


## The loaders to be exposed --------------------------------------------------- 
    

def load_iris():
    return skdatasets.load_iris()
    
def load_iriska():
    return _load_local('iriska')

def load_wine():
    return skdatasets.load_wine()

def load_hartigan():
    return _load_local('hartigan1975')
    
def load_soy_small():
    return _load_local('soy_small')

def load_wbco():
    return _load_local('wbco') 
    

