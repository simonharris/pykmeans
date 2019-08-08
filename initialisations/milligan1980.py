'''
Milligan 1980 Ward-based algorithm

See: The validation of four ultrametric clustering algorithms
https://www.sciencedirect.com/science/article/abs/pii/0031320380900011
'''

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def generate(data, K, opts={}):

    ward = AgglomerativeClustering(n_clusters=K, linkage='ward')
    ward.fit(data)
    
    centroids = np.zeros((K, data.shape[1]))

    for k in range(0, K):
        centroids[k,:] = np.mean(data[ward.labels_==k], axis=0)
        
    return centroids
    
