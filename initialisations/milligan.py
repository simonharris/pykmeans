"""
Milligan 1980 Ward-based algorithm

An examination of the effect of six types of error perturbation on fifteen 
clustering algorithms
https://link.springer.com/article/10.1007/BF02293907
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def generate(data, num_clusters):
    """The common interface"""

    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
    ward.fit(data)

    centroids = np.zeros((num_clusters, data.shape[1]))

    for k in range(0, num_clusters):
        centroids[k, :] = np.mean(data[ward.labels_ == k], axis=0)

    return centroids
    
