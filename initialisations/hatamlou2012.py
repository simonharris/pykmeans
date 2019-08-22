"""
Hatamlou 2012 binary search algorithm

See: In search of optimal centroids on data clustering using a binary search algorithm
https://www.sciencedirect.com/science/article/abs/pii/S0167865512001961, 

TODO:
 - "termination criterion"
 - make sure have correct objective function
 - metrics, incl. F-score (skmetrics.fbeta_score?)
 - unit tests. Shouldn't be too hard as deterministic

"""

import numpy as np

from initialisations.Initialisation import Initialisation
from kmeans import distance_table


class Hatamlou(Initialisation):


    def __init__(self, data, K, opts):
        """Constructor"""
    
        self._max_loops = opts['max_loops']
        super().__init__(data, K, opts)
    
    
    def find_centers(self):
        """Main method"""
    
        # How "big" the initial partitions are eg. 1/3 of the data for Iris
        min_ds, max_ds = self._find_min_max(self._data)
        G = (max_ds - min_ds)/self._K
        
        centroids = []

        for i in range(0, self._K):
            Ci = min_ds + i * G
            centroids.append(Ci)
            
        centroids = np.array(centroids)

        # Initial score
        score = self._objective_function(self._data, centroids)
    
        SSM = np.repeat([np.max(self._data, axis=0)], self._K, axis=0)
            
        for loop in range(0, self._max_loops):
            for i in range(0, self._K):
                for j in range(0, self._num_attrs):
             
                    oldattr = centroids[i][j]
                    centroids[i][j] = oldattr + SSM[i][j]
                    
                    newscore = self._objective_function(self._data, centroids)
                    
                    # If improvement hasn't occurred
                    if newscore >= score:
                        
                        centroids[i][j] = oldattr # reinstate it, I guess?
         
                        if SSM[i][j] < 0:
                            SSM[i][j] = -SSM[i][j]/2
                        else:
                            SSM[i][j] = -SSM[i][j]
                    else :        
                        score = newscore
        
        return centroids
    

    def _find_min_max(self, data):
        """Minimum and maximum values of the whole dataset"""
        return np.min(data, 0), np.max(data, 0)
        
        
    def _objective_function(self, data, centroids):
        '''Sum of intra-cluster distances'''
        
        distances = distance_table(data, centroids)
        
        return np.sum(distances.min(1))


## -----------------------------------------------------------------------------


def generate(data, K, opts):
    """The common interface"""
    
    opts['max_loops'] = 300
    
    htmlu = Hatamlou(data, K, opts)
    return htmlu.find_centers()
    
