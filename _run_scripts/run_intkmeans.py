"""Temp bootstrap file to run Intelligent K-Means"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from datasets import testloader
from initialisations import ikmeans as alg

# Finds 91 centroids
# dataset = testloader._load_local('20_1000_1000_r_1.5_025')
# num_clusters = 20

# Finds 383 centroids
# dataset = testloader._load_local('5_50_1000_r_1_025')
# num_clusters = 5

# This is one that didn't complete on Ceres
# dataset = testloader._load_local('20_2_1000_r_1_024')
# num_clusters = 20

# This one fails with < 20 centroids (19)
dataset = testloader._load_local('20_2_1000_r_1.5_035')
num_clusters = 20

centroids = alg.generate(dataset.data, num_clusters)

print(centroids)
print("There were", len(centroids), "centroids found")
