"""Unit tests for Khan & Ahmad 2004"""

import unittest

import numpy as np

from datasets import testloader
from initialisations import khanahmad2004 as ccia

# Method could be a function
# pylint: disable=R0201


class KhanAhmadTestSuite(unittest.TestCase):
    """Unit tests for Khan & Ahmad 2004"""

    def test_with_iris(self):
        """Crude integration-style test until I can break it down a little.
        Does not trigger the merge stage.
        """

        dataset = testloader.load_iris()

        seeds = ccia.generate(dataset.data, 3)

        # As emitted by the Java
        expected = np.array([
            [5.006, 3.428, 1.462, 0.246],
            [6.85384615, 3.07692308, 5.71538462, 2.05384615],
            [5.88360656, 2.74098361, 4.38852459, 1.43442623]])

        np.testing.assert_array_almost_equal(seeds, expected)

    def _test_with_fossil(self):
        """As used in the paper itself. Trigger the merge stage"""

        dataset = testloader.load_fossil()

        seeds = ccia.generate(dataset.data, 3)

        # Output from the buggy code
        expected = np.array([
            [205.0, 29.0, 11.0, 23.0, 90.0, 805.0],
            [117.03333, 48.3083, 9.3928, 27.01785, 84.9381, 453.47023],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        np.testing.assert_array_almost_equal(seeds, expected, 4)
