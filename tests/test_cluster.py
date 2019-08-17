import unittest

import numpy as np

from cluster import Cluster


class ClusterTestSuite(unittest.TestCase):


    def test_assign_retrieve_1(self):
    
        v = [1, 2, 3]
    
        c = Cluster();
        c.assign(v)
        
        self.assertEqual(c.get_samples(), [v])
        
        
    def test_get_mean_1(self):
    
        v = [1, 2, 3]
    
        c = Cluster();
        c.assign(v)
        
        #self.assertEqual(c.get_mean(), v)
        
    
    def test_assign_retrieve_n(self):
        
        v0 = [1, 2, 3]
        v1 = [3, 2, 1]
        v2 = [4, 5, 6]
    
        c = Cluster();
        c.assign(v0)
        c.assign(v1)
        c.assign(v2)    
        
        self.assertEqual(c.get_samples(), [v0, v1, v2])
        
    
    def test_get_mean_n(self):
    
        v0 = [2, 2, 2]
        v1 = [3, 5, 1]
        v2 = [4, 5, 6]
    
        c = Cluster();
        c.assign(v0)
        c.assign(v1)
        c.assign(v2)    
        
        expected = [3, 4, 3]
        
        np.testing.assert_array_equal(c.get_mean(), expected)
        
     
    def test_get_distance_1(self):
     
        v = [1, 1, 1, 1]
        
        c = Cluster()
        c.assign(v)
        
        self.assertEqual(c.get_distance([2, 2, 2, 2]), 2)
        
        
    def test_merge(self):
    
        a = Cluster()
        a.assign([1, 2, 3])
        a.assign([2, 3, 4])
        
        b = Cluster()
        b.assign([3, 4, 5])
        b.assign([4, 5, 6])
        
        a.merge(b)
        
        # test values (order not important)
        self.assertCountEqual(a.get_samples(), 
                [[1, 2, 3], [4, 5, 6], [2, 3, 4], [3, 4, 5]])
              
        # test recalculated mean
        np.testing.assert_array_equal(a.get_mean(), [2.5, 3.5, 4.5])
        
