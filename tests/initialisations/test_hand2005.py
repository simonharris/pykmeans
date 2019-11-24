"""
Tests for Hand 2005 initialisation algorithm
"""

import unittest

from datasets import loader
from initialisations import hand2005 as hand


class HandTestSuite(unittest.TestCase):
    """Test suite for Hand 2005"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = loader.load_iris()
        centroids = hand.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
