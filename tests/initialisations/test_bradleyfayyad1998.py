import unittest
from initialisations import bradleyfayyad1998 as bf
import numpy as np
import sklearn.datasets as skdatasets
import kmeans


class BfTestSuite(unittest.TestCase):


    def _test_find_furthest(self):
        
        distances = np.array([
            [  1,  2,  3], # 1
            [  7,  5, 16], # 5
            [  7, 26,  4], # 4
            [ 19, 20, 21], # 19
            [  6, 18,  8]  # 6
        ])
        
        np.testing.assert_equal(bf._find_furthest(distances), [3])
        np.testing.assert_equal(np.sort(bf._find_furthest(distances, 2)), [3, 4])
        np.testing.assert_equal(np.sort(bf._find_furthest(distances, 3)), [1, 3, 4])


    def _test_with_1_empty(self):
        '''Seeds and data known to leave one empty cluster after k_means(), and 
        thus trigger k_means_mod() to reassign a centroid'''
        
        seeds = np.array([
            [5.4, 3. , 4.5, 1.5],
            [6.7, 3. , 5. , 1.7],
            [5.1, 3.8, 1.5, 0.3], # Doesn't get any data points assigned
        ])
        
        data = np.array([
            [6.4, 2.9, 4.3, 1.3], # Gets assigned to 0 but is furthest, so becomes the new 2
            [6.3, 3.4, 5.6, 2.4],
            [6.8, 3. , 5.5, 2.1],
            [5. , 2. , 3.5, 1. ],
            [5.8, 2.7, 5.1, 1.9],
        ])
        
        expected_labels = [2, 1, 1, 0, 0]
        
        expected_centroids = [
            [5.4,  2.35, 4.3,  1.45],
            [6.55, 3.2,  5.55, 2.25],
            [6.4,  2.9,  4.3,  1.3 ], # The new 2
        ]

        centroids = bf.k_means_mod(seeds, data, len(seeds))        
        labels = kmeans.distance_table(data, centroids).argmin(1)
        
        np.testing.assert_array_equal(labels, expected_labels)
        np.testing.assert_array_equal(centroids, expected_centroids)
       
    
    def test_with_n_empty(self):
        '''Seeds and data known to leave more than one empty cluster'''
        
        seeds = np.array([
            [6.4, 2.9, 4.3, 1.3], # Gets assigned to 0 but is furthest, so becomes the new 2
            [6.3, 3.4, 5.6, 2.4],
            [6.8, 3. , 5.5, 2.1],
            [5. , 2. , 3.5, 1. ],
            [100, 100, 100, 100],
        ])
        
        data = np.array([
            [6.4, 2.9, 4.3, 1.3], # Gets assigned to 0 but is furthest, so becomes the new 2
            [6.3, 3.4, 5.6, 2.4],
            [6.8, 3. , 5.5, 2.1],
            [5. , 2. , 3.5, 1. ],
            [5.8, 2.7, 5.1, 1.9],
        ])
        
        expected_labels = [2, 1, 1, 0, 0]
        
        expected_centroids = [
            [5.4,  2.35, 4.3,  1.45],
            [6.55, 3.2,  5.55, 2.25],
            [6.4,  2.9,  4.3,  1.3 ], # The new 2
        ]

        centroids = bf.k_means_mod(seeds, data, len(seeds))        
        labels = kmeans.distance_table(data, centroids).argmin(1)
        
        #np.testing.assert_array_equal(labels, expected_labels)
        #np.testing.assert_array_equal(centroids, expected_centroids)
    
