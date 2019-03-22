import unittest
from erisoglu import Erisoglu

class ErisogluTestSuite(unittest.TestCase):

    def setUp(self):
        self._e = Erisoglu()
        self._set_up_data()
        

    def test_find_main_axis(self):  
        self.assertEqual(self._e.main_axis(self._data1), 4)
        self.assertEqual(self._e.main_axis(self._data2), 5)
        
    # TODO   
    #def test_find_secondary_axis(self):
        #self.assertEqual(self._e.secondary_axis(self._data1), 4)
        #self.assertEqual(self._e.secondary_axis(self._data2), 5)
  
    # misc setup methods -------------------------------------------------------
            
    def _set_up_data(self):
    
        self._data1 = [
            [1, 2, 3, 1000, 9],
            [1, 3, 2,  -44, 8],
            [1, 2, 3, 1001, 7],
            [1, 3, 1, -100, 8]
        ]

        self._data2 = [
            [9, 1, 2, 3, 1000, 9],
            [9, 1, 3, 2,  -44, 8],
            [7, 1, 2, 3, 1001, 7],
            [8, 1, 3, 1, -100, 8]
        ]
                
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
    
