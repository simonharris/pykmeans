import numpy as np
import kmeans

def anomalous_pattern(Data):
    '''Locate the most anomalous cluster in a data set'''

    # i) Standardise the original data (idempotent)
    DataStd = kmeans.standardise(Data)

    # ii) Initial setting
    Origin = np.zeros((1, DataStd.shape[1]))

    InitialDist = kmeans.distance_table(DataStd, Origin)
    c = DataStd[InitialDist.argmax()]

    # iii) Cluster update
    while True:
        Z = np.array([Origin[0], c])

        AllDist = kmeans.distance_table(DataStd, Z)
        U = AllDist.argmin(1)
        Ui = np.where(U==1)             # Needed later to remove them from Data
        S = DataStd[U==1, :]

        # iv) Centroid update
        cTentative = np.mean(S, 0)

        if np.array_equal(c, cTentative):
            break
        else:
            c = cTentative

    # v) Output
    return c, S, Ui


def ikmeans(Data):
    '''Intelligent K-Means algorithm'''

    centroids = []

    # ii) Control
    while True:

        # i) Anomalous Pattern
        c, S, Ui = anomalous_pattern(Data)

        centroids.append(c)

        Data = np.delete(Data, Ui, 0)

        print(Data)
        print("Len of data:", len(Data))

        print("================================\n")

        # TODO: investigate other stopping conditions
        if len(centroids) >= 3:
            break

    centroids = np.array(centroids)
    print(centroids)

    # iv) K-Means
    return kmeans.cluster(Data, len(centroids), centroids)


# ------------------------------------------------------------------------------

if __name__ == '__main__':

    data = np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='float')

    '''To test Anomalous Pattern'''
    #c, S, indexes = anomalous_pattern(data)
    #print('Most anomalous centroid:\n', c, "\n")
    #print('Most anomalous cluster:\n', S, "\n")
    #print('Indexes to remove: ', indexes, "\n")

    '''To test I-K-Means'''
    Z, U, clusters, iterations = ikmeans(data)
