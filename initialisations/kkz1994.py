'''
KKZ 1994 algorithm

See: A new initialization technique for generalized Lloyd iteration
https://ieeexplore.ieee.org/abstract/document/329844/
'''

import numpy as np

from kmeans import distance_table


def generate(data, K, opts={}):

    norms = np.linalg.norm(data, axis=1)

    first = data[np.argmax(norms)]
    codebook = np.array([first])

    while codebook.shape[0] < K:
        distances = distance_table(data, codebook)
        mins = np.min(distances, axis=1)
        amax = np.argmax(mins, axis=0)
        nxt = data[amax]
        codebook = np.append(codebook, [nxt], axis=0)
        
    return codebook
    
