
"""
Tests for Likas Global k-means initialisation algorithm
"""

import unittest

from datasets import loader
from initialisations import likas2003 as gkm


class GKMTestSuite(unittest.TestCase):
    """Test suite for Likas/GKM"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = loader.load_iris()
        centroids = gkm.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
