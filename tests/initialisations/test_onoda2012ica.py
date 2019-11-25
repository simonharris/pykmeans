"""
Tests for Onoda 2012 ICA initialisation algorithm
"""

import unittest

from datasets import testloader
from initialisations import onoda2012ica as onoda


class OnodaICATestSuite(unittest.TestCase):
    """Test suite for Onoda/ICA"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = onoda.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
