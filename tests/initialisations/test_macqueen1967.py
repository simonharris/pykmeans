"""Tests for MacQueen 1967 algorithm"""

import unittest

from initialisations import macqueen1967 as macqueen


class MacQueenTestCase(unittest.TestCase):

    def test_get_pairs(self):

        mylist = ['a', 'b', 'c']
        expected = [('a', 'b'), ('a', 'c'), ('b', 'c')]

        pairs = macqueen.get_pairs(mylist)
        self.assertCountEqual(expected, pairs)

        mylist = ['a', 'b', 'c', 'd']
        expected = [('a', 'b'), ('a', 'c'), ('a', 'd'),
                    ('b', 'c'), ('b', 'd'), ('c', 'd')]

        pairs = macqueen.get_pairs(mylist)
        self.assertCountEqual(expected, pairs)
