import unittest

import numpy as np
import sklearn.datasets as skdatasets

from initialisations import khanahmad2004 as ccia


class KhanAhmadTestSuite(unittest.TestCase):

    def test_whole_thing(self):
        """Crude integration-style test until I can break it down a little"""

        iris = skdatasets.load_iris()

        seeds = ccia.generate(iris.data, 3, {})

        expected = np.array([[5.006, 3.428, 1.462, 0.246],
                             [6.85384615, 3.07692308, 5.71538462, 2.05384615],
                             [5.88360656, 2.74098361, 4.38852459, 1.43442623]])

        np.testing.assert_array_almost_equal(seeds, expected)
