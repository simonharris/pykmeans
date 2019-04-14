import numpy as np
import utils as pu
import kmeans

# Intelligent K-Means clustering algorithm
# See: Mirkin 2005 - Clustering for data mining: a data recovery approach

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


def generate(Data, K):
    '''Intelligent K-Means algorithm'''

    DataWorking = Data
    centroids = []

    while True:

        c, S, Ui = anomalous_pattern(DataWorking)

        centroids.append(c)

        DataWorking = np.delete(DataWorking, Ui, 0)

        # TODO: investigate other stopping conditions
        if len(centroids) >= K:
            break

    return np.array(centroids)
