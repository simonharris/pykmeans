"""
Tests for Pavan et al. 2010 SPSS initialisation algorithm
"""

import unittest

import numpy as np

from datasets import loader
from initialisations import spss2010 as spss

# pylint: disable=R0201


class SPSSTestSuite(unittest.TestCase):
    """Test suite for Pavan/SPSS"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = loader.load_iris()
        centroids = spss.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)

    def test_for_first_centroid(self):
        """Ensure we correctly identify the densest point"""

        data = np.array([
            [100, 200, 300, 400],
            [-100, -200, -300, -400],
            [-10, 1, 1, 1],
            [2, 2, 2, 2],
            [30, 3, 3, 3]
            ])

        centroids = spss.generate(data, 1)
        np.testing.assert_array_equal(centroids, np.array([[2, 2, 2, 2]]))
