"""
Tests for standardisation of data
"""

import unittest

import numpy as np

from datasets import testloader
from preprocessors import stddise


class StddiseTestSuite(unittest.TestCase):
    """Tests for my implementation Renato's standardisation formula"""

    def setUp(self):
        self._proc = stddise

    def test_trivial(self):
        """Poor choice of name, but these things take time to take shape"""

        matrix = self._proc.process(self._get_data_trivial())
        ranges = matrix.max(axis=0) - matrix.min(axis=0)

        expected = np.array([1., 1., 1., 1.])
        np.testing.assert_equal(ranges, expected)

    def test_bigger(self):
        """Try an actual dataset"""
        dataset = testloader.load_iris()

        matrix = self._proc.process(dataset.data)
        ranges = matrix.max(axis=0) - matrix.min(axis=0)

        expected = np.array([1., 1., 1., 1.])
        np.testing.assert_equal(ranges, expected)

    @staticmethod
    def _get_data_trivial():
        output = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ]

        return output
