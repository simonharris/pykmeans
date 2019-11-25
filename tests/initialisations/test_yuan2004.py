"""
Tests for Yuan 2004 initialisation
"""

import math
import unittest

import numpy as np
from sklearn.cluster import KMeans

from datasets import testloader
from initialisations import yuan2004 as yuan
from metrics import accuracy

# pylint sees dataset as simply a tuple
# pylint: disable=E1101


class YuanTestSuite(unittest.TestCase):
    """
    Tests for Yuan 2004 initialisation
    """

    def setUp(self):
        self._set_up_data()

    def test_distance_table(self):
        """Test the calculation of distances between all rows"""

        data = self._data1
        table = yuan.distance_table(data)

        # sanity check the shape
        self.assertEqual(table.shape, (len(data), len(data)))
        self.assertTrue(math.isnan(table[1][1]))
        self.assertTrue(math.isnan(table[3][3]))
        self.assertFalse(math.isnan(table[3][2]))
        self.assertFalse(math.isnan(table[3][1]))
        self.assertEqual(table[2][1], 1)
        self.assertAlmostEqual(table[3][2], 13.60147051, places=8)

    def test_find_nearest(self):
        """Test finding the nearest two dazta points in a set"""

        pair = yuan.find_closest(self._data1)
        self.assertEqual(pair, [1, 2])

    def test_find_next_nearest(self):
        """Test finding the nearest point to a set already found"""

        data = self._data1
        group = data[[1, 2]]

        # strictly, 1 and 2 should be deleted from the data at this point, as
        # per the code, but it makes little difference

        nextone = yuan.find_next_closest(data, group)
        self.assertEqual(0, nextone)

        group = np.vstack([group, data[nextone]])
        data = np.delete(data, nextone, 0)

        nextone = yuan.find_next_closest(data, group)
        self.assertEqual(2, nextone)

    def test_find_centroid_basic(self):
        """Test finding centroids in a trivial dataset"""

        num_clusters = 2
        centroids = yuan.generate(self._data1, num_clusters)

        expecteda = [
            (1 + 1) / 2,
            (1 + 1) / 2,
            (2 + 1) / 2,
            (1 + 1) / 2,
        ]
        expectedb = [
            (4 + 7) / 2,
            (4 + 8) / 2,
            (3 + 9) / 2,
            (4 + 7) / 2,
        ]

        self.assertEqual(list(centroids[0]), expecteda)
        self.assertEqual(list(centroids[1]), expectedb)

    def test_against_iris(self):
        """Test run against Iris dataset, as used in the paper"""

        dataset = testloader.load_iris()
        data = dataset.data
        target = dataset.target

        num_clusters = 3

        centroids = yuan.generate(data, num_clusters)

        # sanity check shape
        self.assertEqual(centroids.shape, (num_clusters, 4))

        # run kmeans
        est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
        est.fit(data)

        score = accuracy.score(target, est.labels_)

        # as claimed in paper
        self.assertAlmostEqual(0.886667, score, places=6)

    # Data loading etc --------------------------------------------------------

    def _set_up_data(self):
        self._data1 = np.array([
            [4, 4, 3, 4],
            [1, 1, 2, 1],
            [1, 1, 1, 1],
            [7, 8, 9, 7],
        ])
