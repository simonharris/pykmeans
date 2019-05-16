#from scipy.spatial import distance as spdistance
import numpy as np

'''
Hatamlou 2012 binary search algorithm

See: In search of optimal centroids on data clustering using a binary search algorithm
https://www.sciencedirect.com/science/article/abs/pii/S0167865512001961
'''

def find_min_max(data):
    return np.min(data, 0), np.max(data, 0)
