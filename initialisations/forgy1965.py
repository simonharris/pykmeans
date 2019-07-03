import numpy as np

'''
Forgy 1965 version of random initialisation

See: Cluster Analysis of Multivariate Data: Efficiency vs Interpretability of Classifications
https://www.scirp.org/(S(351jmbntvnsjt1aadkposzje))/reference/ReferencesPapers.aspx?ReferenceID=1785698

'''

def generate(data, K):
    '''Select random data points as initial seeds'''

    return data[np.random.choice(data.shape[0], K, replace=False), :]
    
