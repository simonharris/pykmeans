import unittest
from initialisations.erisoglu import Erisoglu
import numpy as np

class ErisogluTestSuite(unittest.TestCase):

    def setUp(self):
        self._e = Erisoglu()
        self._set_up_data()


    def test_variation_coefficient(self):
        self.assertEqual(self._e.variation_coefficient([1,1,1]), 0)
        self.assertAlmostEqual(self._e.variation_coefficient([1,2,3]), 0.40824829046)
        self.assertAlmostEqual(self._e.variation_coefficient([-1,-2,-3]), 0.40824829046)


    def test_find_main_axis(self):
        self.assertEqual(self._e.find_main_axis(self._data1.T), 3)
        self.assertEqual(self._e.find_main_axis(self._data2.T), 4)


    # Note discrepancy between Erisoglu and Pearson. Currently I've used Pearson
    def test_correlation_coefficient(self):
        self.assertAlmostEqual(self._e.correlation_coefficient([1,2], [2,4]), 1)
        self.assertAlmostEqual(self._e.correlation_coefficient([2,1], [2,4]), -1)
        self.assertAlmostEqual(self._e.correlation_coefficient([1,2,3,4,5], [2,4,6,8,10]), 1)
        self.assertAlmostEqual(self._e.correlation_coefficient([10,2,3,4,5,6,99], [1,2,3,4,3,2,1]), -0.546, 4)


    def test_find_secondary_axis(self):
        self.assertEqual(self._e.find_secondary_axis(self._data1.T, 3), 4)
        self.assertEqual(self._e.find_secondary_axis(self._data2.T, 4), 5)


    def test_find_center(self):

        # Some extra simple data. Nb. assume this already to be transposed
        data = [[  1,    2,   3], # Should be secondary. Mean is 2
                [200, -100, 200], # Should be main. Mean is 100
                [100,  -50, 100]]
        self.assertEqual(self._e.find_center(data, 1, 0), [100, 2])

        self.assertEqual(self._e.find_center(self._data2.T, 4, 5), [1, 8])


    def test_distances(self):
        # Between two points
        self.assertEqual(
            self._e.distance(np.array([-7,-4]), np.array([17,6])),
            26)
        self.assertEqual(
            self._e.distance(np.array([-7,-4]), np.array([17,6])),
            26)
        self.assertAlmostEqual(
            self._e.distance(np.array([1,7,98,56,89]), np.array([8,6,56,5,0])),
            111.0675470)

        # Between n points
        self.assertEqual(
            self._e.distance(np.array([1]), np.array([1]), np.array([1])),
            0)
        self.assertEqual(
            self._e.distance(np.array([0,1]), np.array([1,1]), np.array([1,1])),
            2)
        self.assertEqual(
            self._e.distance(np.array([0,1]), np.array([1,1]), np.array([1,1]), np.array([0,3])),
            4)

        # And by unpacking a list
        mypoints = [np.array([1,1]), np.array([1,1]), np.array([0,3])]
        self.assertEqual(self._e.distance(np.array([0,1]), *mypoints), 4)

    # misc setup methods -------------------------------------------------------

    def _set_up_data(self):

        self._data1 = np.array([
            [0, 190, 3, 1000, 9],
            [1, 200, 2, -999, 8],
            [1, 190, 3, 1001, 7],
            [1, 189, 1, -998, 8]
        ])

        self._data2 = np.array([
            [9, 0, 2, 3, 1000, 9],
            [9, 1, 3, 2, -999, 8],
            [7, 1, 2, 3, 1001, 7],
            [8, 1, 3, 1, -998, 8]
        ])

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
