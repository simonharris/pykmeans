"""Tests for the Cluster object"""

import unittest

import numpy as np

from cluster import Cluster

# For the methods that use np.testing rather than self
# pylint: disable=R0201


class ClusterTestSuite(unittest.TestCase):
    """Test suite for the Cluster object"""

    def test_assign_retrieve_1(self):
        """Test retrieving a single vector"""

        vect = [1, 2, 3]
        clus = Cluster()
        clus.assign(vect)
        self.assertEqual(clus.get_samples(), [vect])

    def test_get_mean_1(self):
        """Test calculating mean on a single vector"""

        vect = [1, 2, 3]
        clus = Cluster()
        clus.assign(vect)
        np.testing.assert_array_equal(clus.get_mean(), vect)

    def test_assign_retrieve_n(self):
        """Test retrieving multiple vectors"""

        v_0 = [1, 2, 3]
        v_1 = [3, 2, 1]
        v_2 = [4, 5, 6]

        clus = Cluster()
        clus.assign(v_0)
        clus.assign(v_1)
        clus.assign(v_2)

        self.assertEqual(clus.get_samples(), [v_0, v_1, v_2])

    def test_get_mean_n(self):
        """Test calculating mean on multiple vectors"""

        v_0 = [2, 2, 2]
        v_1 = [3, 5, 1]
        v_2 = [4, 5, 6]

        clus = Cluster()
        clus.assign(v_0)
        clus.assign(v_1)
        clus.assign(v_2)

        expected = [3, 4, 3]

        np.testing.assert_array_equal(clus.get_mean(), expected)

    def test_get_distance_1(self):
        """Test Euclidean distance"""

        vect = [1, 1, 1, 1]

        clus = Cluster()
        clus.assign(vect)

        self.assertEqual(clus.get_distance([2, 2, 2, 2]), 2)

    def test_merge(self):
        """Test merging clusters"""

        clus_a = Cluster()
        clus_a.assign([1, 2, 3])
        clus_a.assign([2, 3, 4])

        clus_b = Cluster()
        clus_b.assign([3, 4, 5])
        clus_b.assign([4, 5, 6])

        clus_a.merge(clus_b)

        # test values (order not important)
        self.assertCountEqual(clus_a.get_samples(),
                              [[1, 2, 3], [4, 5, 6], [2, 3, 4], [3, 4, 5]])

        # test recalculated mean
        np.testing.assert_array_equal(clus_a.get_mean(), [2.5, 3.5, 4.5])
