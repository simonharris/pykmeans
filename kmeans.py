import numpy as np

'''
K-Means clustering algorithm

See: Mirkin 2005 - Clustering for data mining: a data recovery approach
'''

def distance_table(Data, Z, columns=None):
    '''Calculate distances between entities (1 per row) and centroids (1 per column)'''

    K = len(Z)

    AllDist = np.zeros((Data.shape[0], K))

    if columns:
        Data = Data[:, columns]
        Z = Z[:, columns]

    for k in range(K):
        AllDist[:, k] = np.sum((Data - Z[k, :])**2, 1)

    return AllDist


def cluster(Data, K, seeds=None):
    '''K-Means clustering algorithm rewritten'''

    N, M = Data.shape

    # Randomly initialise Z unless seeds are supplied
    if seeds is None:
        Z = Data[np.random.choice(N, K, replace=False), :]
    else:
        Z = seeds

    OldU = []  # Indices of previous nearest centroids
    iterations = 0

    # Main loop
    while True:

        iterations += 1

        AllDist = distance_table(Data, Z)

        U = AllDist.argmin(1)

        if (np.array_equal(OldU, U)):
            break

        clusters = []

        # Generate new centroids and clusters
        for k in range(K):
            cluster = Data[U==k, :]
            Z[k, :] = np.mean(cluster, 0)
            clusters.append(cluster)

        OldU = U

    return {'centroids':Z, 'labels':U, 'clusters':clusters, 'iterations':iterations}
