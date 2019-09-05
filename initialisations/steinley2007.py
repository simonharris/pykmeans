'''
Steinley 2003 algorithm

See: Local Optima in K-Means Clustering: What You Don't Know May Hurt You
https://psycnet.apa.org/fulltext/2003-09632-004.html
'''

import math

import numpy as np


def generate(data, K, opts={}):

    N = data.shape[0]

    Z_init = New_Z_init = np.zeros((K, data.shape[1]))

    SSE = math.inf

    for i in range(0, opts['restarts']):

        U = np.random.randint(low=0, high=K, size=N)
        
        EmptyCluster = False;

        NewSSE = 0;

        for k in range(0, K):
            if np.sum(U==k) == 0:
                EmptyCluster = True  
                #print("Empty cluster found")
            else :
            
                centroid = np.mean(data[U==k,:], axis=0);
             
                New_Z_init[k,:] = centroid
                
                NewSSE += np.sum(np.sum((data[U==k,:] - centroid)**2, axis=1))        
        
        if EmptyCluster: 
            continue # goto next restart

        if NewSSE < SSE:
            Z_init = New_Z_init;
            SSE = NewSSE;
            print("SSE is now: %s" % (SSE))

    return Z_init

