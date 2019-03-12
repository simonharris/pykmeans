import numpy as np
from kmeans import distance_table
import utils
from sklearn import preprocessing

def anomolous_pattern(Data):

    # Step i) Standardise the data
    Data2 = utils.get_standardised_matrix(Data)

    print(Data2)
    print("Grand mean:", Data2.mean())
    print("Stdev:", np.std(Data2))
    #print(Data)

    Data3 = preprocessing.scale(Data)

    print(Data3)
    print("Grand mean:", Data3.mean())
    print("Stdev:", np.std(Data3))
    #print(Data)

    return("Hello World")


# ------------------------------------------------------------------------------

if __name__ == '__main__':

    data = np.loadtxt('sample_data/Learning_Data.csv', delimiter=',', dtype='int')

    foo = anomolous_pattern(data)


# ------------------------------------------------------------------------------

'''
[[-1  0  0 ...  1  0  0]
 [ 0  0  0 ...  0  1  0]
 [ 0  0  0 ...  0  0  0]
 ...
 [ 0  0  0 ...  0  0  0]
 [ 1  2  2 ...  1  0  0]
 [-1  0  0 ...  0  0  0]]
Grand mean: 0.20373683164380837
Stdev: 0.6585929943246764
'''