"""Tests for the Dataset object"""

import unittest

from dataset import Dataset


class DatasetTestCase(unittest.TestCase):
    """Tests for Dataset object"""

    def test_accessors(self):
        """Test constructing object and retrieving properties"""

        name = "myname"
        data = [1]
        labels = [2]

        dset = Dataset(name, data, labels)

        self.assertEqual(dset.name, name)
        self.assertEqual(dset.data, data)
        self.assertEqual(dset.target, labels)

    def test_num_clusters(self):
        """Calculate K/the number of clusters"""

        name = "myname"
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        labels = [2, 2, 3, 3, 2, 3, 4, 4, 2, 4]

        dset = Dataset(name, data, labels)

        self.assertEqual(dset.num_clusters(), 3)
