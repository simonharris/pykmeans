import unittest
import numpy as np
from metrics import accuracy

class AccuracyScoreTest(unittest.TestCase):

    def test_accuracy_score(self):

        # Yuan 2004 Iris
        cm = np.array(
                [[ 0,  0, 50],
                 [ 3, 47,  0],
                 [36, 14,  0]])

        self.assertAlmostEqual(accuracy.from_matrix(cm), 0.886667, places=6)

        # Yuan 2004 Wine
        cm = np.array(
                [[31,  1, 27],
                 [ 7, 64,  0],
                 [11, 37,  0]])
        self.assertAlmostEqual(accuracy.from_matrix(cm), 0.685393, places=6)
