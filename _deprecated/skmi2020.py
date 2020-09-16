"""Experimental initialistation algorithm"""

import numpy as np

from initialisations.base import Initialisation
from kmeans import distance_table


class SKMI(Initialisation):

    def _calc_density(self, point, latestdata):
        """Sum of distances to its nearest neighbours"""

        neighbours = int(len(latestdata)/self._num_clusters) + 1
        dists = distance_table(np.array([point]), latestdata)[0]
        idx = np.argpartition(dists, neighbours)
        subdists = dists[idx[:neighbours]]
        return np.sum(subdists)

    def _find_first_centroid(self, latestdata):
        """The first promising point"""

        density = [self._calc_density(point, latestdata)
                   for point in latestdata]
        return latestdata[np.argmin(density)]

    def _find_furthest(self, temp_centroids, latestdata):
        """The furthest-nearest point (exact opposite of Yuan)"""

        distances = distance_table(latestdata, temp_centroids)
        nearests = np.min(distances, axis=1)
        return latestdata[np.argmax(nearests)]

    def find_centers(self):

        centroids = []

        to_find = self._num_clusters
        data = self._data

        while to_find > 1:

            first = self._find_first_centroid(data)
            centroids.append(first)

            temp_centroids = np.array([first])

            while len(temp_centroids) < to_find:
                furthest = self._find_furthest(temp_centroids, data)
                temp_centroids = np.vstack((temp_centroids, furthest))

            # Delete latest
            clustering = np.argmin(
                    distance_table(temp_centroids, data), axis=0)
            mask = np.where(clustering == 0)[0]
            data = np.delete(data, mask, axis=0)

            to_find -= 1

        # Finally just get the mean of the remaining points
        final = np.mean(data, axis=0)

        centroids.append(final)

        return np.array(centroids)


def generate(data, num_clusters):
    """The common interface"""

    alg = SKMI(data, num_clusters)
    return alg.find_centers()
