import numpy as np
from kmeans import distance_table
from sklearn import preprocessing

def anomolous_pattern(Data):

    # i) standardise the original data
    ##DataStd = preprocessing.scale(Data)

    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    DataStd = min_max_scaler.fit_transform(Data)

    #print(DataStd)

    # ii) Initial setting
    N, M = DataStd.shape
    Origin = np.zeros((1, M))

    AllDist = distance_table(DataStd, Origin)
    c = DataStd[AllDist.argmax()]


# ------------------------------------------------------------------------------

if __name__ == '__main__':

    data = np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='float')

    foo = anomolous_pattern(data)
