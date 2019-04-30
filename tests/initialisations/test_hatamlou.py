import unittest
import numpy as np
from initialisations import hatamlou2012 as hatamlou


class HatamlouTestCase(unittest.TestCase):

    def test_min_max(self):
        '''Sanity check that numpy does what I expected'''
        
        data = np.array([[1.0,2.0,3.0],
                         [0.5,3.0,4.0],
                         [2.0,4.0,0.1]])

        min, max = hatamlou.find_min_max(data)
        self.assertListEqual(list(min), [0.5,2.0,0.1])
        self.assertListEqual(list(max), [2.0,4.0,4.0])

