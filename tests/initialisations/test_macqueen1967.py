"""Tests for MacQueen 1967 algorithm"""

import unittest

from datasets import testloader
from initialisations import macqueen1967 as macqueen


class MacQueenTestCase(unittest.TestCase):
    """Tests for MacQueen 1967 algorithm"""

    def test_code_runs(self):
        """This needs more, but at least prove it runs"""

        dataset = testloader.load_iris()
        centroids = macqueen.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)

    def _test_with_hartigan(self):
        """This is left as TODO as it exposes the fact that the algorithm is
        highly sensitive to the values of R and C being specified for each
        dataset. This seems potentially problematic"""

        dataset = testloader.load_hartigan()
        centroids = macqueen.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)

    def test_get_pairs(self):
        """This appears to simply test a Python function, but probably was
        for testing an earlier implementation. No harm keeping it"""

        mylist = ['a', 'b', 'c']
        expected = [('a', 'b'), ('a', 'c'), ('b', 'c')]

        pairs = macqueen.get_pairs(mylist)
        self.assertCountEqual(expected, pairs)

        mylist = ['a', 'b', 'c', 'd']
        expected = [('a', 'b'), ('a', 'c'), ('a', 'd'),
                    ('b', 'c'), ('b', 'd'), ('c', 'd')]

        pairs = macqueen.get_pairs(mylist)
        self.assertCountEqual(expected, pairs)
