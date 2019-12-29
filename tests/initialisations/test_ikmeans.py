"""
Test for Mirkin 2005 Intelligent K-Means algorithm
"""

import unittest

# import numpy as np

from datasets import testloader
from initialisations import ikmeans as ikminit
# import kmeans

# py lint: disable=R0201,W0212


class IkmTestSuite(unittest.TestCase):
    """Test suite for Mirkin/IKM"""

    def setUp(self):
        """Run prior to every test method"""

    def test_code_runs(self):
        """At least prove it runs"""

        dataset = testloader.load_iris()
        centroids = ikminit.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)

    def test_with_hartigan(self):
        """A tiny dataset which can't possibly work here"""

        dataset = testloader.load_hartigan()
        centroids = ikminit.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)
