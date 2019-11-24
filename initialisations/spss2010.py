"""
Pavan et al. 2011 Single Pass Seed Selection (SPSS) algorithm

See: Single pass seed selection algorithm for k-means (2010)
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.8956

and: Robust seed selection algorithm for k-means type algorithms (2011)
https://arxiv.org/abs/1202.1585
"""

import numpy as np

from initialisations.base import Initialisation
from kmeans import distance_table


class SPSS(Initialisation):
    """Single Pass Seed Selection (SPSS) algorithm"""

    def find_centers(self):
        """Main method"""

        centroids = np.array([self._find_hdp()])

        # Remaining required centroids (exactly as per k-means++)
        while len(centroids) < self._num_clusters:

            distances = distance_table(self._data, centroids)
            probabilities = distances.min(1)**2 / np.sum(distances.min(1)**2)

            randindex = np.random.choice(self._num_samples,
                                         replace=False,
                                         p=probabilities)
            centroids = np.append(centroids, [self._data[randindex]], 0)

        return centroids

    def _find_hdp(self):
        """The highest density point"""

        distances = distance_table(self._data, self._data)
        sum_v = np.sum(distances, axis=1)  # doesn't matter which axis
        return self._data[np.argmin(sum_v)]


def generate(data, num_clusters):
    """The common interface"""

    spss = SPSS(data, num_clusters)
    return spss.find_centers()
