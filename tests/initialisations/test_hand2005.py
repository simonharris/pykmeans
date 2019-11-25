"""
Tests for Hand 2005 initialisation algorithm
"""

import unittest

from datasets import testloader
from initialisations import hand2005 as hand


class HandTestSuite(unittest.TestCase):
    """Test suite for Hand 2005"""

    # TODO: something in here is very broken
    def _test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = hand.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
