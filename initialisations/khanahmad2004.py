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
from scipy.special import erfcinv
from sklearn.cluster import KMeans


class CCIA():
    """Cluster Center Initialization Algorithm"""

    def __init__(self, data, K):
        self._data = data
        self._K = K
        self._num_samples = data.shape[0]
        self._num_attrs = data.shape[1]


    def find_centers(self):
        '''Find centers corresponding to each attribute'''

        cluster_string = np.zeros((self._num_samples, self._num_attrs))

        # for each attribute
        for i in range(0, self._num_attrs):
            val = self._data[:, i]

            mystr = self._cluster_numeric_attribute(val, self._data)
            # print(mystr)

            membership = self._generate_cluster_string(mystr, self._data)
            # print(membership)

            for l in range(0, self._num_samples):
                cluster_string[l][i] = membership[l]

        # end for each attribute

        cstr = self._extract_cluster_strings(cluster_string, self._data)
        dist_class_str = self._find_unique_cluster_strings(cstr)

        return self._find_initial_centers(cstr, dist_class_str, self._data)


    # Private methods ---------------------------------------------------------


    def _k_means_clustering(self, data, means, K):
        '''Simple wrapper for K-means'''

        est = KMeans(K, init=means, n_init=1)
        est.fit(data)
        return est.labels_


    def _cluster_numeric_attribute(self, attrib, data):
        '''Run K-means on a single attribute'''

        xs = []

        attr_mean = np.mean(attrib)

        # using non-default ddof=1 gives same as Khan's Java and Gnumeric
        attr_sd = np.std(attrib, ddof=1)

        # print("m=" + str(mn) + " sd=" + str(sd))

        for i in range(0, self._K):
            percentile = (2*(i+1)-1) / (2*self._K)
            z = math.sqrt(2) * erfcinv(2*percentile)
            xs.append(z * attr_sd + attr_mean)

        ad = attrib.reshape(-1, 1)
        seeds = np.array(xs).reshape(-1, 1)

        return self._k_means_clustering(ad, seeds, self._K)


    def _generate_cluster_string(self, mystr, data):
        """
        Find new centers corresponding to this attribute's cluster
        allotments and allot data objects based on cluster allotments

        TODO: this is just calculating means. Vectorise it
        """

        clust = np.zeros((self._K, self._num_attrs))
        count = [0] * self._K

        # for each data point label
        for i in range(0, len(mystr)):

            # for each attribute
            for j in range(0, self._num_attrs):
                clust[mystr[i]][j] += self._data[i][j]

            count[mystr[i]] += 1

        # same loops again to get means
        for i in range(0, self._K):
            for j in range(0, self._num_attrs):
                clust[i][j] = clust[i][j]/count[i]

        return self._k_means_clustering(self._data, clust, self._K)


    def _extract_cluster_strings(self, cluster_string, data):
        '''
        Extract clustering strings for the whole data

        TODO: can be heavily refactored
        '''

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


    def _distinct_attributes(self, args):
        '''Count distinct attribute values'''

        return Counter(args)


    def _find_initial_centers(self, cstr, dist_class_str, data):

        init_centers = np.zeros((len(dist_class_str), data.shape[1]))
        cnt = np.zeros(len(dist_class_str))

        # print(cstr)
        # print(dist_class_str)

        for key, val in enumerate(cstr):
            j = 0

            # for each pairs
            for item in dist_class_str:
                # print(str(item) + " = " + str(dist_class_str[item])
                #      + ' --> ' + str(item == val))

                if item == val:
                    for k in range(0, data.shape[1]):
                        init_centers[j][k] += data[key][k]
                    cnt[j] += 1
                    break
                j += 1

        for i in range(0, len(dist_class_str)):
            for j in range(0, data.shape[1]):
                init_centers[i][j] = init_centers[i][j] / cnt[i]

        # TODO: MergeDBMSDC

        return init_centers


# ----------------------------------------------------------------------------


def generate(data, K, opts):
    '''The common interface'''

    ccia = CCIA(data, K)
    return ccia.find_centers()
