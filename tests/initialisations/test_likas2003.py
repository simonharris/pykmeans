
"""
Tests for Likas Global k-means initialisation algorithm
"""

import unittest

from datasets import testloader
from initialisations import globalkm


class GKMTestSuite(unittest.TestCase):
    """Test suite for Likas/GKM"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = globalkm.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
