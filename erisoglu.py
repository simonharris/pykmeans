import numpy as np
import utils as pu
import scipy as sp

#
# Erisoglu 2011 "new" algorithm:
# See: A new algorithm for initial cluster centers in k-means algorithm
# https://www.sciencedirect.com/science/article/pii/S0167865511002248
#


class Erisoglu():

    def _variation_coefficient(self, vector):
        '''Absolute value of std dev / mean. Data must not be standardised'''
     
        return abs(sp.stats.variation(vector))
   
   
    def main_axis(self, data):
        '''Step i) Find the column/feature with greatest variance'''
        
        data = self._prepare_data(data)
        
        max = 0
        column = None
        
        for j in range(0, len(data)):
            cvj = self._variation_coefficient(data[j])
            
            if cvj > max:
                max = cvj
                column = j
                
        return j
        
        
    #def secondary_axis(self, data):
    
    #    main = self._main_axis(data)
        
    #    data = self._prepare_data(data)
        
    #    for each column etc...
        
        
    def _prepare_data(self, data):
        '''Since we're working with columns, it's simpler if we transpose first'''
    
        return np.array(data).T      
               

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    pass

