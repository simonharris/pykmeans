"""Tests for my attempt at calculating Accuracy Score"""

import unittest

import numpy as np

from metrics import accuracy


class AccuracyScoreTest(unittest.TestCase):
    """Tests for my attempt at calculating Accuracy Score"""

    def test_accuracy_score(self):
        """Test with some trivial but real data"""

        # Yuan 2004 Iris
        conf_matrix = np.array([
            [0, 0, 50],
            [3, 47, 0],
            [36, 14, 0]
            ])

        self.assertAlmostEqual(
            accuracy.from_matrix(conf_matrix),
            0.886667,
            places=6)

        # Yuan 2004 Wine
        conf_matrix = np.array([
            [31, 1, 27],
            [7, 64, 0],
            [11, 37, 0]
            ])

        self.assertAlmostEqual(
            accuracy.from_matrix(conf_matrix),
            0.685393,
            places=6)
