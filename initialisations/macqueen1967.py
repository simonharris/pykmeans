'''
MacQueen 1967 algorithm

See: Some methods for classification and analysis of multivariate observations
https://books.google.co.uk/books?id=IC4Ku_7dBFUC&pg=PA281#v=onepage&q&f=false
'''

from itertools import combinations

import numpy as np
import sklearn.datasets as skdatasets

from cluster import Cluster


R = C = 1.7 # Seems about right for Iris, but other datasets will vary

 
def get_pairs(alist):
   
    return list(combinations(alist, 2))
    

def consolidate(clusters):
    
    # "Until all the means are separated by an amount of C or more"
    while True:
    
        if len(clusters) == 1:
            return clusters

        pairs = get_pairs(clusters)
    
        distances = [pair[0].get_distance(pair[1].get_mean()) for pair in pairs]
        
        if min(distances) > C:
            return clusters

        pair_to_merge = pairs[np.argmin(distances)]

        left = pair_to_merge[0]
        right = pair_to_merge[1]

        left.merge(right)

        del clusters[clusters.index(right)]


def generate(data, K, opts={}):

    clusters = []

    for i in range(0, len(data)):
    
        # create clusters from first K samples
        if i < K:
            c = Cluster()
            c.assign(data[i])
            clusters.append(c)
            continue
        
        clusters = consolidate(clusters)

        # get distance from each cluster
        distances = [c.get_distance(data[i]) for c in clusters]
            
        if min(distances) > R:
            c = Cluster()
            c.assign(data[i])
            clusters.append(c)     
        else:
            # assign to nearest
            clusters[np.argmin(distances)].assign(data[i])
                                
        clusters = consolidate(clusters)
            
    return np.array([cl.get_mean() for cl in clusters])     
    
