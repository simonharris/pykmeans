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

        # The most densely surrounded point is the first initial centroid
        centre_h = self._find_hdp()
        centroids = np.array([centre_h])

        # Find the remaining required centroids
        while len(centroids) < self._num_clusters:

            distances = distance_table(self._data, centroids)
            print(distances)

            mins_D = np.min(distances, axis=1)
            print(mins_D)

            # 6) Find y as ...
            partition = np.partition(mins_D, self._how_many)[:self._how_many]
            print(partition)
            y = sum(partition)
            print("y:", y)

            # 7) Find the unique integer i so that...
            i = 0
            accum_dist = 0

            while accum_dist < y:

                accum_dist = accum_dist + mins_D[i]  # But nb they say ^2?!

                i = i + 1

                print(centroids)
                print(self._data[i])

                print("AD:", accum_dist)

                ##
                ## But surely the i found here isn't a meaningful index to X?
                ## Must be some sort of argmin thing?
                ## Just looks like he's cycling thought the data in a way
                ##   that's highly dependent on the order of the data
                ##

            centroids = np.vstack((centroids, self._data[i]))
            print("Centroids:\n", centroids)

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
