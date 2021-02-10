"""
Tests for Hand 2005 initialisation algorithm
"""

import unittest

from datasets import testloader
from initialisations import hand


class HandTestSuite(unittest.TestCase):
    """Test suite for Hand 2005"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = hand.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)

    def test_with_hartigan(self):
        """A tiny dataset which led to problems with empty clusters"""

        dataset = testloader.load_hartigan()
        centroids = hand.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)
