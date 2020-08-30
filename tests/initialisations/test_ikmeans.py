"""
Test for Mirkin 2005 Intelligent K-Means algorithm

The _deprecated tests could potentially be brought back in some for one day,
but it's not a priority just now.
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import ikm_card as ikminit_c
from initialisations import ikm_first as ikminit_f
from initialisations.base import InitialisationException

# pylint: disable=R0201,W0212


class IkmTestSuite(unittest.TestCase):
    """Test suite for Mirkin/IKM"""

    def test_with_hartigan(self):
        """A tiny dataset which can't possibly work here"""  # why not, Simon?

        dataset = testloader.load_hartigan()
        centroids = ikminit_c.generate(dataset.data, 3)
        self.assertEqual((3, 3), centroids.shape)

    def test_exception_when_it_cant_reach_k(self):
        """Check for exception when it doesn't reach K clusters"""

        dataset = testloader._load_local('20_2_1000_r_1.5_035')
        num_clusters = 20

        with self.assertRaises(InitialisationException):
            ikminit_c.generate(dataset.data, num_clusters)

    def test_card_with_known_output(self):
        """Test I haven't made things worse in introducing abstract class"""

        expected = [[5.658065, 2.645161, 4.145161, 1.267742],
                    [5.006, 3.428, 1.462, 0.246],
                    [6.60333333, 2.98, 5.43166667, 1.94],
                    ]

        data = self._get_test_data()
        num_clusters = 3

        centroids = ikminit_c.generate(data, num_clusters)
        np.testing.assert_allclose(centroids, expected, rtol=1e-6)

    def test_first_with_known_output(self):
        """Test the new selection strategy"""

        # Output from previous implementation, right or wrong
        expected = [[6.60333333, 2.98, 5.43166667, 1.94],
                    [5.006, 3.428, 1.462, 0.246],
                    [5.658065, 2.645161, 4.145161, 1.267742],
                    ]

        data = self._get_test_data()
        num_clusters = 3

        centroids = ikminit_f.generate(data, num_clusters)
        np.testing.assert_allclose(centroids, expected, rtol=1e-6)

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
