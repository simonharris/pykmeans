'''
Forgy 1965 version of random initialisation

At least, I think it is. The paper does not seem to be available and accounts 
are contradictory, ofteen describing Forgy as random data points. However, many 
papers credit that idea to Faber 1994.

See: Cluster Analysis of Multivariate Data: Efficiency vs Interpretability of Classifications
https://www.scirp.org/(S(351jmbntvnsjt1aadkposzje))/reference/ReferencesPapers.aspx?ReferenceID=1785698
'''

import numpy as np


def generate(data, K, opts={}):
    '''Assign each data point to random cluster and return the means'''    

    clusters = np.random.randint(0, K, data.shape[0])
    centroids = np.zeros((K, data.shape[1]))

    for k in range(0, K):
        centroids[k,:] = np.mean(data[clusters==k], axis=0)
        
    return centroids
    
