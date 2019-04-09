import numpy as np
import utils as pu
import kmeans

#
# Intelligent K-Means clustering algorithm
# See: Mirkin 2005, Clustering for data mining: a data recovery approach
#

def anomalous_pattern(Data):
    '''Locate the most anomalous cluster in a data set'''

    # i) Standardise the original data (idempotent)
    Data = pu.standardise(Data)

    # ii) Initial setting
    Origin = np.zeros((1, Data.shape[1]))

    InitialDist = kmeans.distance_table(Data, Origin)
    c = Data[InitialDist.argmax()]

    # iii) Cluster update
    while True:
        Z = np.array([Origin[0], c])

        AllDist = kmeans.distance_table(Data, Z)
        U = AllDist.argmin(1)
        Ui = np.where(U==1)             # Needed later to remove them from Data
        S = Data[U==1, :]

        # iv) Centroid update
        cTentative = np.mean(S, 0)

        if np.array_equal(c, cTentative):
            break
        else:
            c = cTentative

    # v) Output
    return c, S, Ui


def cluster(Data):
    '''Intelligent K-Means algorithm'''

    DataWorking = Data

    centroids = []

    # ii) Control
    while True:

        # i) Anomalous Pattern
        c, S, Ui = anomalous_pattern(DataWorking)

        centroids.append(c)

        DataWorking = np.delete(DataWorking, Ui, 0)

        # TODO: investigate other stopping conditions
        if len(centroids) >= 3:
            break

    centroids = np.array(centroids)
    #print("Initial centroids from ikmeans:\n",  centroids, "\n")

    # iv) K-Means
    return kmeans.cluster(Data, len(centroids), centroids)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    data = np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='float')
    data = pu.standardise(data)

    '''To test Anomalous Pattern'''
    #c, S, indexes = anomalous_pattern(data)
    #print('Most anomalous centroid:\n', c, "\n")
    #print('Most anomalous cluster:\n', S, "\n")
    #print('Indexes to remove: ', indexes, "\n")

    '''To test I-K-Means'''
    Z, U, clusters, iterations = cluster(data)

    print("U:\n", U, "\n")
    print("Centroids:\n", Z, "\n")
    print("Iterations: ", iterations, "\n")
    #for cluster in clusters:
    #    print("Cluster:\n", cluster, "\n")
