import numpy as np
import utils as pu

# K-Means clustering algorithm
# See: Mirkin 2005 - Clustering for data mining: a data recovery approach

def distance_table(Data, Z):
    '''Calculate distances between entities (1 per row) and centroids (1 per column)'''

    K = len(Z)

    AllDist = np.zeros((Data.shape[0], K))

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

    return Z, U, clusters, iterations

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    # Some defaults
    seeds = None
    K = 3

    '''To test using trivial arrays'''
    #data = np.array([[7,2,3], [5,6,10], [4,5,6], [1,9,10]])
    #seeds = np.array([[1,2,3], [1,2,10]])

    '''To test using Learning_Data.csv'''
    data = pu.standardise(np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='float'))
    seeds = pu.standardise(np.array([[9.,  5.,  5.,  4.,  4.,  5.,  4.,  3.,  3.],
                      [1.,  1.,  1.,  1.,  2.,  1.,  2.,  1.,  1.],
                      [9., 10., 10., 10., 10.,  5., 10., 10., 10.]]))

    Z, U, clusters, iterations = cluster(data, K, seeds)

    #for cluster in clusters:
    #    print("Cluster:\n", cluster, "\n")
    print("U:\n", U, "\n")
    print("Centroids:\n", Z, "\n")
    print("Iterations: ", iterations, "\n")
