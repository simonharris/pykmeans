"""
Test for Bradley & Fayyad 1998 initialisation algorithm
"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import bradley as bfinit
import kmeans

# pylint: disable=R0201,W0212


class BfTestSuite(unittest.TestCase):
    """Test suite for B&F"""

    def test_code_runs(self):
        """At least prove it runs"""

        dataset = testloader.load_iris()
        centroids = bfinit.generate(dataset.data, 3)
        self.assertEqual((3, 4), centroids.shape)

    def test_with_hartigan(self):
        """A tiny dataset which can't possibly work here"""

        dataset = testloader.load_hartigan()

        with self.assertRaises(ValueError):
            bfinit.generate(dataset.data, 3)

    def test_find_furthest(self):
        """Find the data point furthest from its cluster center"""

        distances = np.array([
            [1, 2, 3],     # 1
            [7, 5, 16],    # 5
            [7, 26, 4],    # 4
            [19, 20, 21],  # 19
            [6, 18, 8]     # 6
        ])

        np.testing.assert_equal(bfinit._find_furthest(distances), [3])
        np.testing.assert_equal(np.sort(bfinit._find_furthest(distances, 2)),
                                [3, 4])
        np.testing.assert_equal(np.sort(bfinit._find_furthest(distances, 3)),
                                [1, 3, 4])

    def test_with_1_empty(self):
        """Seeds and data known to leave one empty cluster after k_means(),
        and thus trigger k_means_mod() to reassign a centroid"""

        seeds = np.array([
            [5.4, 3.0, 4.5, 1.5],
            [6.7, 3.0, 5.0, 1.7],
            [5.1, 3.8, 1.5, 0.3],  # Doesn't get any data points assigned
        ])

        data = np.array([
            # Assigned to 0 but is furthest, so becomes the new 2
            [6.4, 2.9, 4.3, 1.3],
            [6.3, 3.4, 5.6, 2.4],
            [6.8, 3.0, 5.5, 2.1],
            [5.0, 2.0, 3.5, 1.0],
            [5.8, 2.7, 5.1, 1.9],
        ])

        expected_labels = [2, 1, 1, 0, 0]

        expected_centroids = [
            [5.4, 2.35, 4.3, 1.45],
            [6.55, 3.2, 5.55, 2.25],
            [6.4, 2.9, 4.3, 1.3],     # The new 2
        ]

        centroids = bfinit._k_means_mod(seeds, data, len(seeds))
        labels = kmeans.distance_table(data, centroids).argmin(1)

        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(centroids, expected_centroids)

    def _test_with_n_empty(self):
        """Seeds and data known to leave more than one empty cluster

        This is left as TODO for now, since no way can I force sklearn to
        give me more than one empty cluster.
        """
