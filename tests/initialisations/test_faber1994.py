"""
Tests for Random Centroids algorithm
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import random_c


class RandomCTestSuite(unittest.TestCase):
    """Test suite for Random Centroids"""

    def test_initialisation(self):
        """Prove it runs and sanity check output"""

        num_clusters = 3

        dataset = testloader.load_iris()
        centroids = random_c.generate(dataset.data, num_clusters)
        self.assertEqual((num_clusters, 4), centroids.shape)

        unique = np.unique(centroids, axis=0)
        self.assertEqual(len(unique), num_clusters)
