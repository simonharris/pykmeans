import unittest
from kmeans import distance_table
import numpy as np

class DistanceTableTestSuite(unittest.TestCase):

    #
    # TODO: we don't seem to have taken the sqrt of the sums
    # of squared distances, which we did in the project. Check
    # with Renato
    #
    def test_distance_table(self):

        data = np.array([[1,1], [2,3], [4,4]])
        centroids = np.array([[2,2], [3,3]])
        dtable = distance_table(data, centroids)

        self.assertEqual(dtable.shape, (3,2))

        expected = np.array([[2,8], [1,1], [8,2]])
        self.assertTrue(np.array_equal(dtable, expected))

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
