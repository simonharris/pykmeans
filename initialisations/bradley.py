"""
Bradley & Fayyad 1998

See: Refining Initial Points for K-Means Clustering
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.50.8528&rep=rep1&type=pdf
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings

from initialisations import random as randominit
from kmeans import distance_table

# Called J in the paper. 10 is suggested by Steinley 2007
NUM_SUBSETS = 10


def _refine(seeds, data, num_clusters, num_subsets):
    """Main Refine() algorithm"""

    if len(data) < (num_clusters * num_subsets):
        raise ValueError("Dataset too small to be clustered")

    # The J possible "clustering solutions over the subsamples", CMi
    all_solutions = []

    # All candidate centroids found so far, for "smoothing" later, CM
    all_candidates = []

    # Step 1: collect potential solutions -------------------------------------

    # As per Steinley 2007
    subset_size = int(len(data)/num_subsets)

    for _ in range(0, num_subsets):

        subset = data[np.random.choice(data.shape[0],
                                       subset_size,
                                       replace=False), :]
        centroids = _k_means_mod(seeds, subset, num_clusters)

        all_solutions.append(centroids)

        for cent in centroids:
            all_candidates.append(cent)

    all_candidates = np.unique(all_candidates, axis=0)

    # Step 2: smoothing -------------------------------------------------------

    best = None

    for i in range(0, num_subsets):

        clustering = _k_means(all_solutions[i], all_candidates, num_clusters)

        if best is None or clustering.inertia_ < best.inertia_:
            best = clustering

    return best.cluster_centers_


@ignore_warnings(category=ConvergenceWarning)
def _k_means(seeds, data, num_clusters):
    """Calls the standard k-means with the given seeds.

    Warnings are suppressed as empty clusters are expected behaviour."""

    est = KMeans(n_clusters=num_clusters, init=seeds, n_init=1)
    est.fit(data)
    return est


def _k_means_mod(seeds, subset, num_clusters):
    """The KMeansMod() step of the algorithm"""

    clustering = _k_means(seeds, subset, num_clusters)
    centroids = clustering.cluster_centers_

    # Because labels_ returned by kmeans are arbitrarily numbered,
    # we work with the returned centroids
    distances = distance_table(subset, centroids)
    labels = distances.argmin(axis=1)

    sought = set(range(0, num_clusters))
    labels = set(labels)
    missing = sought - labels

    missingcount = len(missing)

    if missingcount > 0:
        # print("Missing:", missing)

        furthest = _find_furthest(distances, missingcount)
        # print("Furthest-nearest:", furthest)

        i = 0
        for clusterid in missing:
            # print("Replacing", seeds[clusterid], "with", subset[furthest[i]])
            seeds[clusterid] = subset[furthest[i]]
            i += 1

        clustering = _k_means(seeds, subset, num_clusters)
        centroids = clustering.cluster_centers_

    return centroids


def _find_furthest(distances, howmany=1):
    """Find data points which are furthest from their assigned cluster center

    This takes the output from distance_table(). For each point we take the
    shortest distance as the assigned center, then select the largest of those,
    so basically the furthest nearest distance(s)"""

    mins = distances.min(axis=1)

    return np.argpartition(mins, -howmany)[-howmany:]


def generate(data, num_clusters):
    """Provide a consistent interface"""

    # It isn't explicitly stated what this should be, but this is as
    # described by Steinley 2007 and seems a reasonable assumption
    starting_point = randominit.generate(data, num_clusters)

    return _refine(starting_point, data, num_clusters, NUM_SUBSETS)
