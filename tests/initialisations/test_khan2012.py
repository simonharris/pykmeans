"""
Tests for Khan 2012 Seed Selection Algorithm
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import khan

# Access to protected member
# pylint: disable=W0212


class KhanTestSuite(unittest.TestCase):
    """Khan test suite"""

    def setUp(self):
        self._set_up_data()
        self._column = 0

    def test_sort_by_magnitude(self):
        """Test sorting data by magnitude"""

        np.testing.assert_equal(
            khan._sort_by_magnitude(self._data, self._column),
            self._sorted)

    def test_find_distances(self):
        """Test finding distances between data points"""

        distances = khan._find_distances(self._sorted, self._column)
        expected = np.array([3, 7, 5, 2])
        np.testing.assert_equal(distances, expected)

    def test_find_split_points(self):
        """Test finding the split points"""

        distances = khan._find_distances(self._sorted, self._column)
        np.testing.assert_equal(khan._find_split_points(distances, 3), [1, 2])

    def test_find_centroids(self):
        """Test finding the actual centroids"""

        """TODO: difficult to test while the column is chosen at random
        Revisit this if we change that

        centroids = khan.generate(self._data, 3)

        expected = np.array([
            [-0.5, 11, 1.5],
            [5, 3, 2],
            [11, 11, 11],
            ])

        np.testing.assert_equal(centroids, expected)"""

    def test_with_iris(self):
        """At least prove it runs"""

        dataset = testloader.load_iris()
        centroids = khan.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)

    def test_with_hartigan(self):
        """A tiny dataset which led to problems with empty clusters"""

        dataset = testloader.load_hartigan()
        centroids = khan.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)

    # Utilities ---------------------------------------------------------------

    def _set_up_data(self):
        self._data = np.array([
            [10, 10, 10],
            [1, 1, 1],
            [-2, 21, 2],
            [5, 3, 2],
            [12, 12, 12],
            ])

        self._sorted = np.array([
            [1, 1, 1],
            [-2, 21, 2],
            [5, 3, 2],
            [10, 10, 10],
            [12, 12, 12],
            ])
