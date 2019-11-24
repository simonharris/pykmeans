"""
Tests for KKZ in1994 initialisation algorithm
"""

import unittest

import numpy as np

from initialisations import kkz1994 as kkz


class KKZTestSuite(unittest.TestCase):
    """Test for KKZ 194 initialisation algorithm"""

    def setUp(self):
        self._load_data()

    def test_with_1(self):
        """Sanity check it gets the most extreme point"""

        centroids = kkz.generate(self._data, 1)
        np.testing.assert_array_equal(centroids, np.array([self._data[1]]))

    def test_with_3(self):
        """Then find the furthest nearest point"""

        centroids = kkz.generate(self._data, 3)
        np.testing.assert_array_equal(centroids,
                                      np.array([self._data[1],
                                                self._data[4],
                                                self._data[0]])
                                      )

    def _load_data(self):

        self._data = np.array([
            [1, 2, 3, 4],
            [-100, -200, -100, -124],
            [5, 3, 4, 5],
            [1, 3, 4, 5],
            [10, 20, 10, 30],
            ])
