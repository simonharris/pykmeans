import numpy as np

'''
Babu 1993 genetic algorithm

See: A near-optimal initial seed value selection in K-means means algorithm
using a genetic algorithm
https://www.sciencedirect.com/science/article/abs/pii/016786559390058L
'''

def find_bounds(data):

    return np.min(data, 0), np.max(data, 0)
    
