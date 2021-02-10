"""
Likas et al. 1993 Global K-Means algorithm

See: The global k-means clustering algorithm
https://www.sciencedirect.com/science/article/pii/S0031320302000602
"""

import numpy as np
from sklearn.cluster import KMeans


def run_k_means(data, centroids):
    """Runs K means N time to try to find successive centroids"""

    best = None

    for row in data:

        candidates = np.concatenate((centroids, [row]), axis=0)

        estimator = KMeans(n_clusters=len(candidates),
                           init=np.array(candidates), n_init=1)
        estimator.fit(data)

        if best is None or estimator.inertia_ < best.inertia_:
            best = estimator

    return best.cluster_centers_


def generate(data, num_clusters):
    """The common interface"""

    centroids = [np.mean(data, axis=0)]

    while len(centroids) < num_clusters:
        centroids = run_k_means(data, centroids)

    return centroids
