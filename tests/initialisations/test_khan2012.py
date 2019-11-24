"""
Tests for Khan 2012 Seed Selection Algorithm
"""

import unittest

import numpy as np

from initialisations import khan2012 as khan


class KhanTestSuite(unittest.TestCase):
    """Khan test suite"""

    def setUp(self):
        self._set_up_data()
        self._column = 0

    def test_sort_by_magnitude(self):
        """Test sorting data by magnitude"""
        np.testing.assert_equal(
            khan.sort_by_magnitude(self._data, self._column),
            self._sorted)

    def test_find_distances(self):
        """Test finding distances between data points"""

        distances = khan.find_distances(self._sorted, self._column)
        expected = np.array([3, 7, 5])
        np.testing.assert_equal(distances, expected)

    def test_find_split_points(self):
        """Test finding the split points"""

        distances = khan.find_distances(self._sorted, self._column)

        np.testing.assert_equal(khan.find_split_points(distances, 3), [1, 2])

    # Utilities ---------------------------------------------------------------

    def _set_up_data(self):
        self._data = np.array([
            [10, 10, 10],
            [1, 1, 1],
            [-2, 21, 2],
            [5, 3, 2],
            ])

        self._sorted = np.array([
            [1, 1, 1],
            [-2, 21, 2],
            [5, 3, 2],
            [10, 10, 10],
            ])
