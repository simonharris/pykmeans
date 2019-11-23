"""
Milligan 1980 Ward-based algorithm

See: The validation of four ultrametric clustering algorithms
https://www.sciencedirect.com/science/article/abs/pii/0031320380900011
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
