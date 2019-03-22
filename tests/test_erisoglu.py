import unittest
#import numpy as np
import erisoglu as e

class ErisogluTestSuite(unittest.TestCase):

    def test_coefficient_of_variation(self):
        '''Step i) Variation coefficient'''
        
        # Interesting SE discussion: https://bit.ly/2JyDLkN
                
        # stddev is 0 so this will always be 0
        self.assertEqual(e.coefficient_of_variation([4,4,4]), 0)
        
        # 50 / 50
        self.assertEqual(e.coefficient_of_variation([0,100,0,100]), 1)
        
        # Check negatives are abs()ed
        self.assertEqual(e.coefficient_of_variation([0,-100,0,-100]), 1)            

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
    
