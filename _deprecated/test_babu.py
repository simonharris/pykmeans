import unittest
#from initialisations.babu1993 import Babu
from initialisations import babu1993 as babu
import numpy as np
import sklearn.datasets as skdatasets

class BabuTestSuite(unittest.TestCase):

    def test_find_bounds(self):

        data = np.array([
            [1, 2, 3, 4],
            [4, 3, 1, 1],
            [5, 7, 3, 5]
        ])

        L, R = babu.find_bounds(data);
        np.testing.assert_equal(L, np.array([1, 2, 1, 1]))
        np.testing.assert_equal(R, np.array([5, 7, 3, 5]))
