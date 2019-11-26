"""
Tests for Hatamlou 2012 algorithm

Plenty to be done here
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import hatamlou2012 as htm
from initialisations.hatamlou2012 import Hatamlou

# Allow testing protected method
# pylint: disable=W0212


class HatamlouTestCase(unittest.TestCase):
    """Tests for Hatamlou 2012 algorithm"""

    def test_min_max(self):
        """Sanity check that numpy does what I expected"""

        data = np.array([[1.0, 2.0, 3.0],
                         [0.5, 3.0, 4.0],
                         [2.0, 4.0, 0.1]])

        htmlu = Hatamlou(data, 3)

        mins, maxes = htmlu._find_min_max(data)
        self.assertListEqual(list(mins), [0.5, 2.0, 0.1])
        self.assertListEqual(list(maxes), [2.0, 4.0, 4.0])

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = htm.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)
