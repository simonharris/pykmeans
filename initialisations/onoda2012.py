import numpy as np
from sklearn.decomposition import FastICA, PCA

'''
Onoda 2012 ICA-based algorithm

See: Careful seeding method based on independent components analysis for k-means clustering
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.663.5343&rep=rep1&type=pdf#page=53
'''

def _run_pca(data, K):
    '''Run Pricipal Component Analysis'''

    pca = PCA(n_components=K)
    pca.fit(data)
    return pca.components_


def _run_ica(data, K):
    '''Run Independent Component Analysis'''
    
    # Compute ICA
    #ica = FastICA(n_components=K) # return 3x4
    #ica = FastICA() # returns 4x4 - completely different numbers
    #S_ = ica.fit_transform(data)  # Reconstruct signals
    #A_ = ica.mixing_  # Get estimated mixing matrix

    #print(ica.components_) # The unmixing matrix


def _find_centroids(data, components):
    '''Step 1b from the algorithm'''

    C = []

    for component in components:
        distances = [_calc_distance(x, component) for x in data]
        C.append(data[np.argmin(distances)])

    return np.array(C)


def _calc_distance(row, component):
    '''Used in Step 1b from the algorithm'''
    
    def mag(vector):
        return np.linalg.norm(vector)
    
    return np.dot(component, row) / (mag(component) * mag(row))


# Main interface ---------------------------------------------------------------

    
def generate(data, K, opts={'method':'PCA'}):
    '''Provide consistent interface'''

    #if (opts['method'] == 'PCA'):
    #    
    components = _run_pca(data, K)
    #else:
    #    components = _run_ica(data, K)
    #print(components)
    
    return _find_centroids(data, components)
        
        
