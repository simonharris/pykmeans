"""
Onoda 2012 ICA-based algorithm

See: Careful seeding method based on independent components analysis for
 k-means clustering
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.663.5343&rep=rep1&type=pdf#page=53
"""

import numpy as np
from sklearn.decomposition import FastICA, PCA


def _run_pca(data, num_components):
    """Run Pricipal Component Analysis"""

    pca = PCA(n_components=num_components)
    pca.fit(data)
    return pca.components_


def _run_ica(data, num_components):
    """Run Independent Component Analysis"""

    ica = FastICA(n_components=num_components)
    ica.fit_transform(data)
    return ica.components_


def _find_centroids(data, components):
    """Step 1b from the algorithm"""

    centroids = []

    for component in components:
        distances = [_calc_distance(x, component) for x in data]
        centroids.append(data[np.argmin(distances)])

    return np.array(centroids)


def _calc_distance(row, component):
    """Used in Step 1b from the algorithm"""

    mag = np.linalg.norm

    return np.dot(component, row) / (mag(component) * mag(row))


# Main interface --------------------------------------------------------------


def generate(data, K, opts={'method': 'ICA'}):
    """Provide consistent interface"""

    if opts['method'] == 'PCA':
        components = _run_pca(data, K)
    else:
        components = _run_ica(data, K)
    # print(components)

    return _find_centroids(data, components)
