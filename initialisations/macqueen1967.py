'''MacQueen 1967'''

import numpy as np
import sklearn.datasets as skdatasets

from cluster import Cluster


def generate(data, K, opts={}):

    clusters = []

    for i in range(0, len(data)):
    
        # create clusters from first K samples
        if i < K:
            c = Cluster()
            c.assign(data[i])
            clusters.append(c)
            
        # then assign the rest to whichever they're nearest to
        else:
            
            # get distance for each cluster
            distances = [c.get_distance(data[i]) for c in clusters]
            
            # assign to nearest
            clusters[np.argmin(distances)].assign(data[i])
            
    return np.array([cl.get_mean() for cl in clusters])
    
