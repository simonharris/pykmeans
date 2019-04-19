import unittest
from initialisations.erisoglu import Erisoglu
import numpy as np
import sklearn.datasets as skdatasets

class ErisogluTestSuite(unittest.TestCase):

    def setUp(self):
        self._e = Erisoglu()
        self._set_up_data()

    # Test a few calculation functions -----------------------------------------

    def test_variation_coefficient(self):
        self.assertEqual(self._e.variation_coefficient([1,1,1]), 0)
        self.assertAlmostEqual(self._e.variation_coefficient([1,2,3]), 0.40824829046)
        self.assertAlmostEqual(self._e.variation_coefficient([-1,-2,-3]), 0.40824829046)

    # Note discrepancy between Erisoglu and Pearson. Currently I've used Pearson
    def test_correlation_coefficient(self):
        self.assertAlmostEqual(self._e.correlation_coefficient([1,2], [2,4]), 1)
        self.assertAlmostEqual(self._e.correlation_coefficient([2,1], [2,4]), -1)
        self.assertAlmostEqual(self._e.correlation_coefficient([1,2,3,4,5], [2,4,6,8,10]), 1)
        self.assertAlmostEqual(self._e.correlation_coefficient([10,2,3,4,5,6,99], [1,2,3,4,3,2,1]), -0.546, 4)

    def test_distances(self):
        # Between two points
        self.assertEqual(self._e.distance([-7,-4], [17,6]), 26)
        self.assertAlmostEqual(self._e.distance([1,7,98,56,89], [8,6,56,5,0]), 111.0675470)

        # Between n points
        self.assertEqual(self._e.distance([0,1], [1,1], [1,1]), 2)
        self.assertEqual(self._e.distance([0,1], [1,1], [1,1], [0,3]), 4)

        # And by unpacking a list
        mypoints = [[1,1], [1,1], [0,3]]
        self.assertEqual(self._e.distance([0,1], *mypoints), 4)

    # Test the actual algorithm ------------------------------------------------

    def test_generate_1(self):
        K = 1
        centroids = self._e.generate(self._data1, K)
        self.assertEqual(len(centroids), K)
        self.assertListEqual(list(centroids[0]), [1, 190, 3, 1001, 7])

    def test_generate_2(self):
        K = 2
        centroids = self._e.generate(self._data1, K)
        self.assertEqual(len(centroids), K)
        self.assertListEqual(list(centroids[0]), [1, 190, 3, 1001, 7])
        self.assertListEqual(list(centroids[1]), [1, 200, 2, -999, 8])

    def test_generate_3(self):
        K = 3
        centroids = self._e.generate(self._data1, K)
        self.assertEqual(len(centroids), K)
        self.assertListEqual(list(centroids[0]), [1, 190, 3, 1001, 7])
        self.assertListEqual(list(centroids[1]), [1, 200, 2, -999, 8])
        self.assertListEqual(list(centroids[2]), [0, 190, 3, 1000, 9])
        ## TODO: the fourth one isn't what you'd hope it to be...

    def XXtest_iris(self):
        K = 3
        dataset = skdatasets.load_iris()
        data = dataset.data
        centroids = self._e.generate(data, K)



    # misc setup methods -------------------------------------------------------

    def _set_up_data(self):

        # Center is [1,8] - means of columns 3 and 4
        self._data1 = np.array([
            [0, 190, 3, 1000, 9], # Furthest from ...,-999,8 so third centroid
            [1, 200, 2, -999, 8], # Furthest from ...,1001,7 so second centroid
            [1, 190, 3, 1001, 7], # Furthest from ...,1,8 so initial centroid
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
