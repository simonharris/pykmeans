"""Unit tests for home-made k-means implementation"""

import unittest

import numpy as np
import sklearn.cluster as skcluster
import sklearn.datasets as skdatasets

from initialisations import erisoglu
import kmeans as mykm


class DistanceTableTestSuite(unittest.TestCase):
    """Unit tests for home-made k-means implementation"""

    def test_distance_table(self):
        """Calculate matrix of distances between two sets of data points"""

        data = np.array([[1, 1], [2, 3], [4, 4]])
        centroids = np.array([[2, 2], [3, 3]])
        dtable = mykm.distance_table(data, centroids)

        self.assertEqual(dtable.shape, (3, 2))

        expected = np.array([[2, 8], [1, 1], [8, 2]])
        self.assertTrue(np.array_equal(dtable, expected))

    def test_vs_sklearn(self):
        """Compare results with scikit-learn implementation"""

        data = skdatasets.load_iris().data
        num_clusters = 3

        # Use Erisoglu as it is deterministic
        seeds = erisoglu.generate(data, num_clusters)

        mine = mykm.cluster(data, num_clusters, seeds)

        theirs = skcluster.KMeans(n_clusters=num_clusters,
                                  n_init=1,
                                  init=seeds)
        theirs.fit(data)

        # Assert same centroids
        np.testing.assert_array_almost_equal(mine['centroids'],
                                             theirs.cluster_centers_,
                                             decimal=6)

        # Assert SSE calculated correctly
        self.assertAlmostEqual(mine['inertia'], theirs.inertia_, places=8)
