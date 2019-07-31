'''General purpose cluster class'''

import numpy as np
from scipy.spatial import distance as spdistance


class Cluster():

    def __init__(self):
        self._mean = None
        self._samples = []

    
    def assign(self, vector):
        self._samples.append(vector)
        self._calculate_mean()
    
    
    def get_samples(self):
        return self._samples
    
    
    def get_mean(self):
        return self._mean
        
    
    def get_distance(self, vector):
        return spdistance.euclidean(self.get_mean(), vector)
        
        
    def _calculate_mean(self):
        self._mean = np.mean(np.array(self._samples), axis=0)

