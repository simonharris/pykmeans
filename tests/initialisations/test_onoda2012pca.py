"""
Tests for Onoda 2012 PCA initialisation algorithm
"""

import unittest

from datasets import loader
from initialisations import onoda2012pca as onoda


class OnodaICATestSuite(unittest.TestCase):
    """Test suite for Onoda/PCA"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = loader.load_iris()
        centroids = onoda.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
