"""
MacQueen 1967 algorithm

See: Some methods for classification and analysis of multivariate observations
https://books.google.co.uk/books?id=IC4Ku_7dBFUC&pg=PA281#v=onepage&q&f=false
"""

from itertools import combinations

import numpy as np

from cluster import Cluster

# Seem about right for Iris, but other datasets will vary
ROUGHENING = 1.7
COARSENING = 1.7


def get_pairs(alist):
    """Return all potential paired combinations of list items"""

    return list(combinations(alist, 2))


def consolidate(clusters):
    """Merge any clusters whose centers are too close"""

    # "Until all the means are separated by an amount of C or more"
    while True:

        if len(clusters) == 1:
            return clusters

        pairs = get_pairs(clusters)

        distances = [pair[0].get_distance(pair[1].get_mean())
                     for pair in pairs]

        # print("Closest:", min(distances))

        if min(distances) > COARSENING:
            return clusters

        # print("\t==> Consolidate merging")

        pair_to_merge = pairs[np.argmin(distances)]

        left = pair_to_merge[0]
        right = pair_to_merge[1]

        left.merge(right)

        del clusters[clusters.index(right)]
        # print("Num clusters is now:", len(clusters))


def generate(data, num_clusters):
    """The common interface"""

    clusters = []

    # For each sample in the dataset
    for index, sample in enumerate(data):

        # Create initial clusters from first K samples
        if index < num_clusters:
            # print("Creating cluster:", index)
            clust = Cluster()
            clust.assign(sample)
            clusters.append(clust)
            continue

        # "If the distance between the members of this pair..."
        clusters = consolidate(clusters)

        # "In addition, as each new point is processed..."

        # Get distance from each cluster
        distances = [c.get_distance(sample) for c in clusters]

        if min(distances) > ROUGHENING:
            # print("Adding a new cluster")
            clust = Cluster()
            clust.assign(sample)
            clusters.append(clust)
            # print("Num clusters is now:", len(clusters))
        else:
            # Assign to nearest
            clusters[np.argmin(distances)].assign(sample)

        clusters = consolidate(clusters)

    return np.array([cl.get_mean() for cl in clusters])
