"""
Random Centroids initialisation

Credited by Steinley 2077 to Faber 1994, but it has been around a great deal
longer than that.

See: Clustering and the continuous k-means algorithm
https://www.semanticscholar.org/paper/Clustering-and-the-Continuous-k-Means-Algorithm-Faber/94ab7d7cef96a447d15c81ac3b9f1134575785b2
"""

import numpy as np


def generate(data, num_clusters):
    """Select random data points as initial seeds"""

    indices = np.random.choice(data.shape[0], num_clusters, replace=False)
    return data[indices, :]
