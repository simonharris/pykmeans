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


def refine(seeds, data, K, J):
    """Main algorithm"""

    # The J possible solutions
    CMI = []

    # All the points found so far
    CM = []

    # TODO: this is according to Steinley
    sample_size = int(len(data)/J)

    for i in range(0, J):

        sample = data[np.random.choice(data.shape[0],
                                       sample_size,
                                       replace=False), :]
        centroids = k_means_mod(seeds, sample, K)

        CMI.append(centroids)
        for c in centroids:
            CM.append(c)

    CM = np.unique(CM, axis=0)

    best = None

    for i in range(0, J):

        km = k_means(CMI[i], CM, K)

        if best is None or km.inertia_ < best.inertia_:
            best = km

    return best.cluster_centers_


@ignore_warnings(category=ConvergenceWarning)
def k_means(seeds, data, K):
    """Calls the standard k-means with the given seeds"""

    est = KMeans(n_clusters=K, init=seeds, n_init=1)
    est.fit(data)
    return est


def k_means_mod(seeds, sample, K):
    """In progress"""

    while True:

        km = k_means(seeds, sample, K)

        centroids = km.cluster_centers_

        # Because labels_ returned by kmeans are arbitrarily numbered
        distances = distance_table(sample, centroids)

        labels = distances.argmin(1)

        sought = set(range(0, K))
        labels = set(labels)
        missing = sought - labels

        missingcount = len(missing)

        if missingcount == 0:
            break
        else:
            print("Missing:", missing)
            print("Before:", seeds)
            print("Sample:", sample)

            furthest = _find_furthest(distances, missingcount)

            print("Furthest-nearest:", furthest)

            i = 0
            for clusterid in missing:
                seeds[clusterid] = sample[furthest[i]]
                i += 1

            print("After:", seeds)

    return centroids


def generate(data, K, opts={}):
    """Provide a consistent interface"""

    # TODO: check this is the correct type of random
    seeds = randominit.generate(data, K)

    return refine(seeds, data, K, opts['J'])


def _find_furthest(distances, howmany=1):
    """The largest smallest one"""

    print("Distances:\n", distances)

    mins = distances.min(1)

    print("Mins:\n", mins)

    return np.argpartition(mins, -howmany)[-howmany:]
