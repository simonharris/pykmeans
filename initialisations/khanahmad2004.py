"""
Khan & Ahmad 2004 "Cluster Center Initialization Algorithm"

See: Cluster center initialization algorithm for K-means clustering
https://www.sciencedirect.com/science/article/abs/pii/S0167865504000996

Heavily inspired by the author's own Java implementation:
https://github.com/titubeta/ccia/
"""

from collections import Counter
import math

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.special import erfcinv
from sklearn.cluster import KMeans

from initialisations.base import Initialisation


class CCIA(Initialisation):
    """Cluster Center Initialization Algorithm"""

    _NN = 1  # No idea. See the Java source code

    def find_centers(self):
        '''Find centers corresponding to each attribute'''

        cluster_string = np.zeros((self._num_samples, self._num_attrs))

        # for each attribute
        for i in range(0, self._num_attrs):
            val = self._data[:, i]

            mystr = self._cluster_numeric_attribute(val)
            # print("CNA:", mystr)

            membership = self._generate_cluster_string(mystr)
            # print("GCS: ", membership)

            for l in range(0, self._num_samples):
                cluster_string[l][i] = membership[l]
        # end for each attribute

        cstr = self._extract_cluster_strings(cluster_string)
        # print("ECS:", cstr)
        dist_class_str = self._find_unique_cluster_strings(cstr)

        return self._find_initial_centers(cstr, dist_class_str, self._data)

    # Private methods ---------------------------------------------------------

    @staticmethod
    def _k_means_clustering(data, means, num_clusters):
        '''Simple wrapper for K-means'''

        est = KMeans(num_clusters, init=means, n_init=1)
        est.fit(data)
        return est.labels_

    def _cluster_numeric_attribute(self, attrib):
        """Run K-means on a single attribute"""

        xs = []

        attr_mean = np.mean(attrib)

        # using non-default ddof=1 gives same as Khan's Java and Gnumeric
        attr_sd = np.std(attrib, ddof=1)

        # print("m=" + str(mn) + " sd=" + str(sd))

        for i in range(0, self._num_clusters):
            percentile = (2*(i+1)-1) / (2*self._num_clusters)
            z = math.sqrt(2) * erfcinv(2*percentile)
            xs.append(z * attr_sd + attr_mean)

        ad = attrib.reshape(-1, 1)
        seeds = np.array(xs).reshape(-1, 1)

        return self._k_means_clustering(ad, seeds, self._num_clusters)


    def _generate_cluster_string(self, mystr):
        """
        Find new centers corresponding to this attribute's cluster
        allotments and allot data objects based on cluster allotments

        TODO: this is just calculating means. Vectorise it
        """

        clust = np.zeros((self._num_clusters, self._num_attrs))
        count = [0] * self._num_clusters

        # for each data point label
        for i in range(0, len(mystr)):

            # for each attribute
            for j in range(0, self._num_attrs):
                clust[mystr[i]][j] += self._data[i][j]

            count[mystr[i]] += 1

        # same loops again to get means
        for i in range(0, self._num_clusters):
            for j in range(0, self._num_attrs):
                clust[i][j] = clust[i][j]/count[i]

        return self._k_means_clustering(self._data, clust, self._num_clusters)

    def _extract_cluster_strings(self, cluster_string):
        """
        Extract clustering strings for the whole data

        TODO: can be heavily refactored
        """

        cstr = []

        for i in range(0, self._num_samples):
            cstr.append('')

            for j in range(0, self._num_attrs-1):
                cstr[i] = cstr[i] + str(int(cluster_string[i][j])) + ','

            cstr[i] += str(int(cluster_string[i][self._num_attrs-1]))

        return cstr

    def _find_unique_cluster_strings(self, cstr):
        '''Not sure why this method exists just to call another...'''

        return self._distinct_attributes(cstr)

    @staticmethod
    def _distinct_attributes(args):
        """Count distinct attribute values"""

        return Counter(args)

    def _find_initial_centers(self, cstr, dist_class_str, data):

        init_centers = np.zeros((len(dist_class_str), data.shape[1]))
        cnt = np.zeros(len(dist_class_str))

        for i in range(0, len(cstr)):
            # print("i:", i)
            j = 0

            # for each pairs
            for key in dist_class_str:
                # print("Key:", key)
                # print("foo")
                # print(str(item) + " = " + str(dist_class_str[item])
                #      + ' --> ' + str(item == val))

                if key == cstr[i]:
                    for k in range(0, data.shape[1]):
                        init_centers[j][k] += data[i][k]
                    cnt[j] += 1
                    break

                j += 1

        for i in range(0, len(dist_class_str)):
            for j in range(0, data.shape[1]):
                init_centers[i][j] = init_centers[i][j] / cnt[i]

        # TODO: MergeDBMSDC
        if len(dist_class_str) == self._num_clusters:
            return init_centers
        else:
            # print("TODO: merge algorithm")
            return self._merge_dbmsdc(init_centers, dist_class_str, data)

    def _merge_dbmsdc(self, init_centers, dist_class_str, data):
        """A nasty hybrid of the original Java, and my best guess
        at how to fix the bugs in it"""

        init_centers = init_centers[init_centers[:, 0].argsort()]

        centers = []

        B = list(range(0, len(dist_class_str)))

        for L in range(0, self._num_clusters):

            #print("\n\nLooping for L ==", L)

            R = np.zeros(len(B))

            for i in range(0, len(B)):

                distance = np.zeros(len(B))

                for j in range(0, len(B)):
                    distance[j] = euclidean(init_centers[i], init_centers[j])

                dist_sort = sorted(distance)
                # print("\nDISTANCES:\n", dist_sort)
                R[i] = dist_sort[self._NN]

            minR = min(R)
            # print("Min R:", minR)

            index = 0

            for i in range(0, len(R)):
                if R[i] == minR:
                    index = i
                    break

            S = []

            # print("Will compare to:", init_centers[index])

            to_remove = []
            for i in range(0, len(B)):

                dist = euclidean(init_centers[index], init_centers[i])
                # print("Dist:", dist)

                if dist < (1.5 * minR):
                    S.append(init_centers[B[i]])
                    #print("Adding", init_centers[B[i]], "to S")
                    to_remove.append(i)

            # print("TR:", to_remove)
            B = np.delete(B, to_remove, axis=0)

            # print("S: ", S)
            # print("B: ", B)

            centers.append(np.mean(S, axis=0))

        return centers


# -----------------------------------------------------------------------------


def generate(data, num_clusters):
    """The common interface"""

    ccia = CCIA(data, num_clusters)
    return ccia.find_centers()
