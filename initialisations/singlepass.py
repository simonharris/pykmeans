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

    def __init__(self, data, num_clusters):

        self._how_many = int(len(data) / num_clusters)
        super().__init__(data, num_clusters)

    def find_centers(self):
        """Main method"""

        # 1-3) The most densely surrounded point is the first initial centroid
        centre_h = self._find_hdp()

        # 4) Add X_h to C as the first centroid
        centroids = np.array([centre_h])

        # Find the remaining required centroids
        while len(centroids) < self._num_clusters:

            # 5) For each point xi, set D(xi)...
            distances = distance_table(self._data, centroids)
            mins_d = np.min(distances, axis=1)

            # 6) Find y as ...
            # Though why it's supposedly recalculated on each loop is puzzling
            dist_h = distance_table(np.array([centre_h]), self._data)[0]
            dist_h = dist_h[dist_h != 0]            # Anderson skips the 0 one
            partition = np.partition(dist_h, self._how_many)[:self._how_many]
            my_y = sum(partition)

            # 7-8) Find the unique integer i so that...
            i = 0
            accum_dist = 0

            while accum_dist < my_y:

                accum_dist = accum_dist + mins_d[i]
                i = i + 1

            # 9) Add X_i to C
            # But surely the i found here isn't a meaningful index to X?
            # It just looks like we're cycling thought the data in a way
            # that's highly dependent on its arbitrary order
            centroids = np.vstack((centroids, self._data[i]))

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
