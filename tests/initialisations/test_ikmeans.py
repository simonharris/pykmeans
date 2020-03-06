"""
Test for Mirkin 2005 Intelligent K-Means algorithm

The _deprecated tests could potentially be brought back in some for one day,
but it's not a priority just now.
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import ikmeans_card as ikminit_c
from initialisations import ikmeans_first as ikminit_f
from initialisations.base import InitialisationException

# pylint: disable=R0201,W0212


class IkmTestSuite(unittest.TestCase):
    """Test suite for Mirkin/IKM"""

    def test_with_hartigan(self):
        """A tiny dataset which can't possibly work here"""

        dataset = testloader.load_hartigan()
        centroids = ikminit_c.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)

    def _deprecated_test_find_origin(self):
        """The function doesn't do a great deal, but it represents a step in
        the published algorithm, so could be considered documentation"""

        data = self._get_test_data()
        origin = ikminit_c._find_origin(data)
        expected = [5.84333333, 3.05733333, 3.758, 1.19933333]
        self._assert_close_enough(origin, expected)

    def _deprecated_test_most_distant_basic(self):
        """Find the furthest point from the origin in the toy data"""

        data = self._get_toy_data()

        furthest, _ = ikminit_c._find_most_distant(data)
        np.testing.assert_array_equal(furthest, data[4])

    def _deprecated_test_most_distant_iris(self):
        """Find the furthest point from the origin in Iris"""

        data = self._get_test_data()
        furthest, _ = ikminit_c._find_most_distant(data)

        # [7.7, 2.6, 6.9, 2.3]
        np.testing.assert_array_equal(furthest, data[118])

    def _deprecated_test_anomalous_pattern_1(self):
        """Test the AP as a whole where anomalous cluster has 1 point"""

        data = self._get_toy_data()

        centroid, partition = ikminit_c._anomalous_pattern(data)

        np.testing.assert_array_equal(centroid, [10, 11, 12, 13])
        np.testing.assert_array_equal(partition, [4])

    def _deprecated_test_anomalous_pattern_n(self):
        """Test the AP as a whole where anomalous cluster has n points"""

        data = np.vstack((self._get_toy_data(),
                          [8, 9, 10, 11],
                          [9, 10, 11, 12]))

        centroid, partition = ikminit_c._anomalous_pattern(data)

        np.testing.assert_array_equal(centroid, [9, 10, 11, 12])
        np.testing.assert_array_equal(partition, [4, 5, 6])

    def test_exception_when_it_cant_reach_k(self):
        '''Check for exception when it doesn't reach K clusters'''

        dataset = testloader._load_local('20_2_1000_r_1.5_035')
        num_clusters = 20

        with self.assertRaises(InitialisationException):
            ikminit_c.generate(dataset.data, num_clusters)

    def test_card_with_known_output(self):
        """Test I haven't made things worse in introducing abstract class"""

        # Output from previous implementation, right or wrong
        expected = [[4.66470588, 3.04705882, 1.41176471, 0.2],
                    [5.81081081, 2.73513514, 4.19189189, 1.3],
                    [6.60333333, 2.98, 5.43166667, 1.94],
                    ]

        data = self._get_test_data()
        num_clusters = 3

        centroids = ikminit_c.generate(data, num_clusters)
        np.testing.assert_allclose(centroids, expected, rtol=1e-8)

    def test_first_with_known_output(self):
        """Test the new selection strategy"""

        # Output from previous implementation, right or wrong
        expected = [[6.60333333, 2.98, 5.43166667, 1.94],
                    [5.81081081, 2.73513514, 4.19189189, 1.3],
                    [5., 2.4, 3.2, 1.03333333],
                    ]

        data = self._get_test_data()
        num_clusters = 3

        centroids = ikminit_f.generate(data, num_clusters)
        np.testing.assert_allclose(centroids, expected, rtol=1e-8)

        # Bugfix
        # Can't believe there isn't a nicer syntax though...
        self.assertEqual(str(type(centroids)), "<class 'numpy.ndarray'>")

    # Helper methods ----------------------------------------------------------

    def _get_test_data(self):
        """Fetch some data to test with"""

        dataset = testloader.load_iris()
        return dataset.data

    def _get_toy_data(self):
        """Return a trivial array"""

        return np.array([
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 4, 2, 4],
            [3, 2, 3, 2],
            [10, 11, 12, 13]
        ])

    def _assert_close_enough(self, actual, desired):
        """Saves repetition"""

        np.testing.assert_array_almost_equal(actual, desired, decimal=8)
