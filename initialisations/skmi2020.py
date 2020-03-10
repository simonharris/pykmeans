'''Just a little experiment...'''

import numpy as np
from scipy.spatial import distance as spdistance

from initialisations.base import Initialisation
from kmeans import distance_table


class SKMI(Initialisation):

    def find_centers(self):

        my_data = self._data.copy()
        centroids = []

        # For each cluster
        while len(centroids) < (self._num_clusters - 1):

            # Find the most densely surrounded point remaining
            cent = self._find_hdp(my_data)
            # print("My next point is", cent)
            centroids.append(cent)

            # print("I HAVE", len(centroids), "centroids")
            # print(centroids)

            # Find some stuff we don't want in this cluster by building
            # a list of distractions...
            how_many_faraway_ones = self._num_clusters - len(centroids)
            # print("I NEED", how_many_faraway_ones, "distractions")
            temp_cents = np.array([cent])
            for faraway_one_ctr in range(0, how_many_faraway_ones):

                im_scared_of = np.mean(temp_cents, axis=0)  # check this...
                # print("Current mean:", im_scared_of)

                scary_distances = [spdistance.euclidean(some_point,
                                                        im_scared_of)
                                   for some_point in my_data]

                #   print("SDs:\n", scary_distances)

                # The point in X that's furthest from our temp mean
                my_scary_id = np.argmax(scary_distances)
                scary_point = my_data[my_scary_id]
                temp_cents = np.vstack((temp_cents, scary_point))

                # print("Temp cents:\n", temp_cents)

            # OK, which do they get assigned to...
            # INV: you could run kmeans at this point instead
            my_dist_table = distance_table(temp_cents, my_data)
            # print("MDT:\n", my_dist_table)

            clustering = np.argmin(my_dist_table, axis=0)
            # print("Partition:\n", clustering)

            # delete stuff assigned to from my_data
            assigned_ones = np.argwhere(clustering == 0)
            # print("Ass ones:", assigned_ones)
            # print("Ass ones len:", len(assigned_ones))
            # print(assigned_ones)
            my_data = np.delete(my_data, assigned_ones, axis=0)

            # print("My data is now of length:", len(my_data))

        # Final cluster is mean of what's remaining...
        # though this is entirely otional...try both
        # INV: you could just run the hdp again...
        final_mean = np.mean(my_data, axis=0)
        # print("FM:", final_mean)
        centroids.append(final_mean)

        return np.array(centroids)

    def _find_hdp(self, data):
        """The highest density point"""

        distances = distance_table(data, data)
        sum_v = np.sum(distances, axis=1)  # doesn't matter which axis
        return self._data[np.argmin(sum_v)]


def generate(data, num_clusters):
    """The common interface"""

    skmi = SKMI(data, num_clusters)
    return skmi.find_centers()
