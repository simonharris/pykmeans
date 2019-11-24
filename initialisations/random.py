"""
Forgy 1965 version of random initialisation

At least, I think it is. The paper does not seem to be available and accounts
are contradictory, often describing Forgy as random data points. However, many
papers credit that idea to Faber 1994. Steinley 2003 uses this method.

Celebi 2013 states this waas Forgy's approach and cites Anderberg 1973.
Pena et al directly contradicts this and cites Anderberg 1973!

See: Cluster Analysis of Multivariate Data: Efficiency vs Interpretability
Classifications
https://www.scirp.org/(S(351jmbntvnsjt1aadkposzje))/reference/ReferencesPapers.aspx?ReferenceID=1785698
"""

import numpy as np

from initialisations.base import EmptyClusterException


def generate(data, num_clusters):
    """
    The common interface.

    Assign each data point to random cluster and return the means
    """

    partition = []

    # prevent empty clusters
    ctr = 0
    while len(np.unique(partition)) < num_clusters:

        ctr += 1

        # prevent infinite loops in tiny data
        if ctr >= 100:
            raise EmptyClusterException("Empty clusters could not be avoided")

        partition = np.random.randint(0, num_clusters, data.shape[0])

    centroids = np.zeros((num_clusters, data.shape[1]))

    for k in range(0, num_clusters):
        cluster = data[partition == k]
        centroids[k, :] = np.mean(cluster, axis=0)

    return centroids
