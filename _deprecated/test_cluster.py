import unittest
import numpy as np
from cluster import Cluster


class ClusterTestCase(unittest.TestCase):

    def test_sum_distances(self):

        centroid = np.array([1,2,3])

        data = np.array([[1,2,3], [1,2,3]])
        cluster = Cluster(centroid, data)
        self.assertEqual(cluster.sum_distances(), 0)

        data = np.array([[1,2,3], [1,2,4]])
        cluster = Cluster(centroid, data)
        self.assertEqual(cluster.sum_distances(), 1)

        data = np.array([[1,2,3], [0,0,0], [9,8,7]])
        cluster = Cluster(centroid, data)
        self.assertAlmostEqual(cluster.sum_distances(), 14.511987)
