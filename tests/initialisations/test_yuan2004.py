"""
Tests for Yuan 2004 initialisation
"""

import math
import unittest

import numpy as np

from initialisations import yuan2004 as yuan


class YuanTestSuite(unittest.TestCase):

    def setUp(self):
        self._set_up_data()

    def test_distance_table(self):
        data = self._data1
        table = yuan.distance_table(data)

        # sanity check the shape
        self.assertEqual(table.shape, (len(data), len(data)))
        self.assertTrue(math.isnan(table[1][1]))
        self.assertTrue(math.isnan(table[3][3]))
        self.assertFalse(math.isnan(table[3][2]))
        self.assertFalse(math.isnan(table[4][1]))
        self.assertEqual(table[3][1], 1)
        self.assertEqual(table[5][2], 4)

    def _test_find_nearest(self):
        pair = yuan.find_closest(self._data1)
        self.assertEqual(pair, [1, 3])

    def _test_find_next_nearest(self):
        data = self._data1
        pair = data[[1, 3]]
        self.assertEqual(yuan.find_next_closest(data, pair), 0)

    # TODO: run against larger data
    def _test_find_centroid_basic(self):
        num_clusters = 2
        centroids = yuan.generate(self._data1, num_clusters)

        expecteda = [
            (4 + 1 + 1) / 3,
            (4 + 1 + 1) / 3,
            (3 + 2 + 1) / 3,
            (4 + 1 + 1) / 3,
        ]
        expectedb = [
            (7 + 7 + 5) / 3,
            (7 + 8 + 5) / 3,
            (7 + 9 + 5) / 3,
            (7 + 7 + 5) / 3,
        ]

        self.assertEqual(list(centroids[0]), expecteda)
        self.assertEqual(list(centroids[1]), expectedb)

    # Data loading etc --------------------------------------------------------

    def _set_up_data(self):
        self._data1 = np.array([
            [4, 4, 3, 4],
            [1, 1, 2, 1],
            [7, 7, 7, 7],
            [1, 1, 1, 1],
            [7, 8, 9, 7],
            [5, 5, 5, 5],
            [100, 100, 100, 100]
        ])
