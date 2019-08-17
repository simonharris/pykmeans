import unittest

import numpy as np

from initialisations import macqueen1967 as macqueen


class SteinleyTestCase(unittest.TestCase):

    def test_get_pairs(self):
        
        mylist = ['a', 'b', 'c', 'd']
        expected = [['a', 'b'], ['a', 'c'], ['a', 'd'], 
                    ['b', 'c'], ['b', 'd'], ['c', 'd']]
        
        pairs = macqueen.get_pairs(mylist)
        self.assertCountEqual(expected, pairs)
        
