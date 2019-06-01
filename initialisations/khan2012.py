import numpy as np
from scipy.spatial import distance as spdistance

'''
Khan 2012 initial seed selection algorithm

See: An initial seed selection algorithm for k-means clustering of georeferenced data to improve replicability of cluster assignments for mapping application
https://www.sciencedirect.com/science/article/pii/S1568494612003377
'''

def sort_by_magnitude(data):
    ''' Step a) Sort data by magnitude'''
    idxs = np.argsort(np.linalg.norm(data, axis=1))

    return data[idxs]


def find_distances(data):
    '''Step b) Find Euclidean distances between sorted rows'''

    distances = []

    for i in range(0, len(data)-1):
        distances.append(spdistance.euclidean(data[i], data[i+1]))

    return np.array(distances)


def find_split_points(distances, K):
    '''Step c) Find the largest distances between adjacent rows'''

    howmany = K-1

    return distances.argsort()[-howmany:][::-1]
