"""
Hand 2006 ICA-based algorithm

See: Optimising k-means clustering results with standard software packages
https://www.sciencedirect.com/science/article/pii/S0167947304002038
"""

import numpy as np
from sklearn.cluster import KMeans

# Paper is vague on "a given starting point" so use uniformly random
from initialisations import faber1994 as faber

# As per the paper
OPTS = {'loops': 100, 'alpha': 0.83, 'beta': 0.95}


def _run_k_means(data, num_clusters, seeds=None):
    """Simulate the Matlab interface a little"""

    if seeds is None:
        seeds = faber.generate(data, num_clusters)

    est = KMeans(n_clusters=num_clusters, n_init=1, init=seeds)
    est.fit(data)

    return est.labels_, est.cluster_centers_, est.inertia_


def generate(data, num_clusters):
    """The common interface"""

    # Find a reasonable starting point
    U, z_init, sse = _run_k_means(data, num_clusters)

    alpha = OPTS['alpha']
    beta = OPTS['beta']
    loops = OPTS['loops']

    for i in range(0, loops):

        r = np.random.random_sample(len(U),)

        for index in np.where(r < alpha)[0]:
            OtherValuesOfK = np.where(range(0, num_clusters) != U[index])[0]
            U[index] = np.random.choice(OtherValuesOfK)

        empty_cluster = False

        new_z_init = np.zeros(z_init.shape)

        for k in range(0, num_clusters):
            if sum(U == k) == 0:
                empty_cluster = True
                break

            new_z_init[k] = np.mean(data[U == num_clusters], axis=0)

        if empty_cluster:
            continue

        NewU, _, new_sse = _run_k_means(data, num_clusters, new_z_init)

        if new_sse < sse:
            sse = new_sse
            U = NewU
            z_init = new_z_init

        alpha = alpha * beta

    return z_init
