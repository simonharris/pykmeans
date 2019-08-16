'''
Hand 2006 ICA-based algorithm

See: Optimising k-means clustering results with standard software packages
https://www.sciencedirect.com/science/article/pii/S0167947304002038
'''

import numpy as np
from sklearn.cluster import KMeans

from datasets import loader
from initialisations import faber1994 as faber


def _run_k_means(data, K, seeds=None):
    '''Simulate the Matlab interface a little'''
    
    if seeds is None:
        seeds = faber.generate(data, K, {})
    
    est = KMeans(n_clusters=K, n_init=1, init=seeds)
    est.fit(data)
    
    return est.labels_, est.cluster_centers_, est.inertia_


def generate(data, K, opts={'loops':100, 'alpha':0.3, 'beta':0.95}):
    '''Provide standard initialisation interface'''
    
    # Find a reasonable starting point
    U, Z_init, SSE = _run_k_means(data, K)
    
    alpha = opts['alpha']
    beta = opts['beta']
    loops = opts['loops']

    for i in range(0, loops):

        r = np.random.random_sample(len(U),)
     
        for index in np.where(r<alpha)[0]:
            OtherValuesOfK = np.where(range(0, K) != U[index])[0]
            U[index] = np.random.choice(OtherValuesOfK)    
         
        EmptyCluster = False
        
        New_Z_init = np.zeros(Z_init.shape)
        
        for k in range(0, K):
            if (sum(U==k) == 0):
                EmptyCluster = True
                break
            
            New_Z_init[k] = np.mean(data[U==k], axis=0)
            
        if EmptyCluster:
            continue
            
        NewU, _, NewSSE = _run_k_means(data, K, New_Z_init)
        
        if NewSSE < SSE:
            print(NewSSE)
            SSE = NewSSE
            U = NewU
            Z_init = New_Z_init
            
        alpha = alpha * beta

    return Z_init

