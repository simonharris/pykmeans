import unittest
import numpy as np
from metrics import accuracy

class AccuracyScoreTest(unittest.TestCase):

    def test_accuracy_score(self):

        cm = np.array(
                [[ 0,  0, 50],
                 [ 3, 47,  0],
                 [36, 14,  0]])

        self.assertEqual(accuracy.from_matrix(cm), 0.8866666666666667)
