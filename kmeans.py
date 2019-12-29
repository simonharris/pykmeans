"""
K-Means clustering algorithm

See: Mirkin 2005 - Clustering for data mining: a data recovery approach
"""

import numpy as np


def distance_table(data, centroids, columns=None):
    """Distances between entities (1 per row) and centroids (1 per column)"""

    num_clusters = len(centroids)

    distances = np.zeros((data.shape[0], num_clusters))

    if columns:
        data = data[:, columns]
        centroids = centroids[:, columns]

    for k in range(num_clusters):
        distances[:, k] = np.sum((data - centroids[k, :])**2, 1)

    return distances


def cluster(data, num_clusters, seeds=None):
    """K-Means clustering algorithm"""

    num_samples = data.shape[0]

    # Randomly initialise Z unless seeds are supplied
    if seeds is None:
        seeds = data[np.random.choice(num_samples,
                                      num_clusters,
                                      replace=False), :]

    old_partition = []  # Indices of previous nearest centroids
    iterations = 0

    # Main loop
    while True:

        iterations += 1

        all_dist = distance_table(data, seeds)

        partition = all_dist.argmin(1)

        if np.array_equal(old_partition, partition):
            break

        clusters = []

        # Generate new centroids and clusters
        for k in range(num_clusters):
            cluster_k = data[partition == k, :]

            seeds[k, :] = np.mean(cluster_k, 0)
            clusters.append(cluster_k)

        old_partition = partition

    return {'centroids': seeds,
            'labels': partition,
            'clusters': clusters,
            'iterations': iterations,
            'inertia': np.sum(all_dist.min(1)),
            }
