import numpy as np
from kmeans import distance_table
from sklearn import preprocessing

def anomolous_pattern(Data):
    '''Locate the most anomolous cluster in a data set'''

    # i) Standardise the original data
    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    DataStd = min_max_scaler.fit_transform(Data)

    # ii) Initial setting
    Origin = np.zeros((1, DataStd.shape[1]))

    InitialDist = distance_table(DataStd, Origin)
    c = DataStd[InitialDist.argmax()]

    # iii) Cluster update
    iterations = 0

    while True:
        Z = np.array([Origin[0], c])

        AllDist = distance_table(DataStd, Z)
        U = AllDist.argmin(1)
        S = DataStd[U==1, :]

        # iv) Centroid update
        cTentative = np.mean(S, 0)

        iterations += 1

        if np.array_equal(c, cTentative):
            break
        else:
            c = cTentative

    # v) Output
    return c, S, iterations

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    data = np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='float')

    c, S, iterations = anomolous_pattern(data)

    print('Most anomolous centroid:\n', c, "\n")
    print('Most anomolous cluster:\n', S, "\n")
    print('Iterations: ', iterations, "\n")
