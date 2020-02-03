"""
Tests for standardisation of data
"""

import unittest

import numpy as np

from preprocessors import stddise


class StddiseTestSuite(unittest.TestCase):
    """Tests for my implementation Renato's standardisation formula"""

    def setUp(self):
        self._proc = stddise
        self._data = self._get_data()

    def test_foo(self):
        """Poor choice of name, but these things take time to take shape"""

        matrix = self._proc.process(self._data)
        stddised = matrix.max(axis=0) - matrix.min(axis=0)

        expected = np.array([1., 1., 1., 1.])
        np.testing.assert_equal(stddised, expected)

    @staticmethod
    def _get_data():
        output = [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
        ]

        return output
