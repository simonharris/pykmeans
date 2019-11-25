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
