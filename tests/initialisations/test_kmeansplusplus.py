"""
Tests for k-means++ initialisation algorithm
"""

import unittest

import numpy as np

from initialisations import kmpp


class KMPPTestSuite(unittest.TestCase):
    """Test suite for random initialisation"""

    def test_code_runs(self):
        """There isn't really much more you can test here"""

        data = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7]
            ])

        centroids = kmpp.generate(data, 2)

        self.assertEqual((2, 4), centroids.shape)
