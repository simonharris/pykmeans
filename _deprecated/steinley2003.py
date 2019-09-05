'''
Steinley 2003 algorithm

See: Local Optima in K-Means Clustering: What You Don't Know May Hurt You
https://psycnet.apa.org/fulltext/2003-09632-004.html
'''

import math

import numpy as np
from sklearn.cluster import KMeans

from initialisations import forgy1965 as forgy


def generate(data, K, opts={'loops':5000}):

    SSE = math.inf
    centroids = None

    for i in range(0, opts['loops']):
        
        seeds = forgy.generate(data, K, {})
        
        est = KMeans(n_clusters=K, n_init=1, init=seeds)
        est.fit(data)
        
        if est.inertia_ < SSE:
            SSE = est.inertia_
            centroids = est.cluster_centers_
        
    return np.array(centroids)

