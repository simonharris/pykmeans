"""
Tests for Faber 1994 random algorithm
"""

import unittest

import numpy as np

from datasets import loader
from initialisations import faber1994 as faber


class FaberTestSuite(unittest.TestCase):
    """Test suite for Faber 1994"""

    def test_initialisation(self):
        """Prove it runs and sanity check output"""

        num_clusters = 3

        dataset = loader.load_iris()
        centroids = faber.generate(dataset.data, num_clusters)
        self.assertEqual((num_clusters, 4), centroids.shape)

        unique = np.unique(centroids, axis=0)
        self.assertEqual(len(unique), num_clusters)
