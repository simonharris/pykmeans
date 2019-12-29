"""
Test for Mirkin 2005 Intelligent K-Means algorithm
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import ikmeans as ikminit

# pylint: disable=R0201,W0212


class IkmTestSuite(unittest.TestCase):
    """Test suite for Mirkin/IKM"""

    def test_code_runs(self):
        """At least prove it runs"""

        centroids = ikminit.generate(self._get_test_data(), 3)
        self.assertEqual((3, 4), centroids.shape)

    def test_with_hartigan(self):
        """A tiny dataset which can't possibly work here"""

        dataset = testloader.load_hartigan()
        centroids = ikminit.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)

    def test_standardise(self):
        """This is to test we're using the appropriate numpy function, rather
        than testing the numpy functions itself"""

        standardised = ikminit._standardise(self._get_test_data())

        means = np.mean(standardised, axis=0)
        np.testing.assert_array_almost_equal(means, [0, 0, 0, 0], decimal=12)

        dev = np.std(standardised, axis=0)
        np.testing.assert_array_almost_equal(dev, [1, 1, 1, 1], decimal=12)

    def test_find_origin(self):
        """The function doesn't do a great deal, but it represents a step in
        the published algorithm, so could be considered documentation"""

        data = self._get_test_data()
        origin = ikminit._find_origin(data)
        expected = [5.84333333, 3.05733333, 3.758, 1.19933333]
        self._assert_close_enough(origin, expected)

        data = ikminit._standardise(data)
        origin = ikminit._find_origin(data)
        expected = [0, 0, 0, 0]
        self._assert_close_enough(origin, expected)

    def test_most_distant_basic(self):
        """Find the furthest point from the origin in the toy data"""

        data = self._get_toy_data()

        furthest, _ = ikminit._find_most_distant(data)
        np.testing.assert_array_equal(furthest, data[4])

        # Sanity check it survives standardisation
        data = ikminit._standardise(data)
        furthest, _ = ikminit._find_most_distant(data)
        np.testing.assert_array_equal(furthest, data[4])

    def test_most_distant_iris(self):
        """Find the furthest point from the origin in Iris"""

        data = self._get_test_data()
        furthest, _ = ikminit._find_most_distant(data)

        # [7.7, 2.6, 6.9, 2.3]
        np.testing.assert_array_equal(furthest, data[118])

    def test_anomalous_pattern_1(self):
        """Test the AP as a whole where anomalous cluster has 1 point"""

        data = self._get_toy_data()

        centroid, partition = ikminit._anomalous_pattern(data)

        np.testing.assert_array_equal(centroid, [10, 11, 12, 13])
        np.testing.assert_array_equal(partition, [4])

    def test_anomalous_pattern_n(self):
        """Test the AP as a whole where anomalous cluster has n points"""

        data = np.vstack((self._get_toy_data(),
                          [8, 9, 10, 11],
                          [9, 10, 11, 12]))

        centroid, partition = ikminit._anomalous_pattern(data)

        np.testing.assert_array_equal(centroid, [9, 10, 11, 12])
        np.testing.assert_array_equal(partition, [4, 5, 6])

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
