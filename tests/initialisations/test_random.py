"""
Tests for 'random' initialisation algorithm
"""

import unittest

import numpy as np

from initialisations import random
from initialisations.base import EmptyClusterException


class RandomTestSuite(unittest.TestCase):
    """Test suite for random initialisation"""

    def test_code_runs(self):
        """There isn't really much more you can test here"""

        data = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7]
            ])

        centroids = random.generate(data, 2)

        self.assertEqual((2, 4), centroids.shape)

    def test_exception_generated(self):
        """It should throw an Exception if empty clusters unavoidable"""

        data = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            ])

        with self.assertRaises(EmptyClusterException):
            random.generate(data, 4)
