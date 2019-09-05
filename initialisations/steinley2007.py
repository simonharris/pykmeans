"""
Steinley 2007 algorithm

See: Initializing K-means Batch Clustering: A Critical Evaluation...
https://link.springer.com/article/10.1007/s00357-007-0003-0

Which references, but presents a slightly different algorithm to:

Local Optima in K-Means Clustering: What You Don't Know May Hurt You
https://psycnet.apa.org/fulltext/2003-09632-004.html
"""

import math

import numpy as np

from initialisations.base import Initialisation


class Steinley(Initialisation):
    """Steinley 2007 algorithm"""

    def find_centers(self):
        """Main method"""

        z_init = new_z_init = np.zeros((self._num_clusters, self._num_attrs))

        sse = math.inf

        for _ in range(0, self._opts['restarts']):

            labels = np.random.randint(low=0,
                                       high=self._num_clusters,
                                       size=self._num_samples)

            empty_cluster = False

            new_sse = 0

            for k in range(0, self._num_clusters):
                if np.sum(labels == k) == 0:
                    empty_cluster = True

                else:
                    centroid = np.mean(self._data[labels == k, :], axis=0)
                    new_z_init[k, :] = centroid
                    new_sse += np.sum(np.sum(
                        (self._data[labels == k, :] - centroid)**2,
                        axis=1))

            if empty_cluster:
                continue  # goto next restart

            if new_sse < sse:
                z_init = new_z_init
                sse = new_sse

        return z_init


# -----------------------------------------------------------------------------


def generate(data, num_clusters, opts):
    """The common interface"""

    init = Steinley(data, num_clusters, opts)
    return init.find_centers()
