"""
Tests for Milligan 1980 Ward-based algorithm
"""

import unittest

from datasets import testloader
from initialisations import milligan


class MilliganTestSuite(unittest.TestCase):
    """Test suite for Milligan 1980"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = milligan.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
