"""
Hatamlou 2012 binary search algorithm

See: In search of optimal centroids on data clustering using a binary search
  algorithm
https://www.sciencedirect.com/science/article/abs/pii/S0167865512001961,

TODO:
 - "termination criterion"
 - make sure have correct objective function
 - metrics, incl. F-score (skmetrics.fbeta_score?)
 - unit tests. Shouldn't be too hard as deterministic
"""

import numpy as np

from initialisations.base import Initialisation
from kmeans import distance_table


class Hatamlou(Initialisation):
    """Hatamlou 2012 initialisation algoprithm"""

    _max_loops = 1

    def find_centers(self):
        """Main method"""

        # How "big" the initial partitions are eg. 1/3 of the data for Iris
        min_ds, max_ds = self._find_min_max(self._data)
        G = (max_ds - min_ds)/self._num_clusters

        centroids = []

        for i in range(0, self._num_clusters):
            Ci = min_ds + i * G
            centroids.append(Ci)

        centroids = np.array(centroids)

        # Initial score
        score = self._objective_function(self._data, centroids)

        ssm = np.repeat([np.max(self._data, axis=0)],
                        self._num_clusters,
                        axis=0)

        for _ in range(0, self._max_loops):
            for i in range(0, self._num_clusters):
                for j in range(0, self._num_attrs):

                    oldattr = centroids[i][j]
                    centroids[i][j] = oldattr + ssm[i][j]

                    newscore = self._objective_function(self._data, centroids)

                    # If improvement hasn't occurred
                    if newscore >= score:

                        centroids[i][j] = oldattr  # reinstate it, I guess?

                        if ssm[i][j] < 0:
                            ssm[i][j] = -ssm[i][j]/2
                        else:
                            ssm[i][j] = -ssm[i][j]
                    else:
                        score = newscore

        return centroids

    @staticmethod
    def _find_min_max(data):
        """Minimum and maximum values of the whole dataset"""
        return np.min(data, 0), np.max(data, 0)

    @staticmethod
    def _objective_function(data, centroids):
        """Sum of intra-cluster distances"""

        distances = distance_table(data, centroids)

        return np.sum(distances.min(1))


# -----------------------------------------------------------------------------


def generate(data, num_clusters):
    """The common interface"""

    opts['max_loops'] = 300

    htmlu = Hatamlou(data, num_clusters)
    return htmlu.find_centers()
