import unittest
from initialisations import khan2012 as khan
import numpy as np
import sklearn.datasets as skdatasets

class BabuTestSuite(unittest.TestCase):

    def setUp(self):
        self._setUpData()


    def test_sort_by_magnitude(self):
        np.testing.assert_equal(khan.sort_by_magnitude(self._data), self._sorted)


    def test_find_distances(self):
        distances = khan.find_distances(self._sorted)
        expected = np.array([1.7320508075688772, 3.1622776601683795, 11.74734012447073])
        np.testing.assert_equal(distances, expected)

    def test_find_split_points(self):
        distances = khan.find_distances(self._sorted)

        np.testing.assert_equal(khan.find_split_points(distances, 3), [2, 1])


    # Utilities ----------------------------------------------------------------

    def _setUpData(self):
        self._data = np.array([
                [10,10,10],
                [ 1, 1, 1],
                [ 2, 2, 2],
                [ 5, 3, 2],
            ])

        self._sorted = np.array([
                [ 1, 1, 1],
                [ 2, 2, 2],
                [ 5, 3, 2],
                [10,10,10],
            ])

