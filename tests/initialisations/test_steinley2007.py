"""
Basic tests for Steinley 2007 initialisation

Due to the random nature of the algorithm, testing precsents certain
challenges. So for now just check the code runs and returns vaguely sane output
"""

import unittest

import sklearn.datasets as skdatasets

from initialisations import steinley2007 as steinley


class SteinleyTestSuite(unittest.TestCase):

    def setUp(self):
        pass

    def test_basic(self):

        iris = skdatasets.load_iris()

        K = 3

        opts = {'restarts': 50}
        centroids = steinley.generate(iris.data, K, opts)

        self.assertEqual(centroids.shape, (3, 4))
