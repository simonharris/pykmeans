"""Temp bootstrap file to run K&A"""
import pathhack

from datasets import testloader
from initialisations import erisoglu2011 as alg

from sklearn.cluster import KMeans


# Didn't initially complete on Ceres
# dataset = testloader._load_local('20_2_1000_r_1.5_035')
# num_clusters = 20

# Test for Renato's question
dataset = testloader._load_local('5_2_1000_r_1.5_010')
num_clusters = 5

# Exceptions on Ceres
# dataset = testloader._load_local('wbco')
# num_clusters = 2

data = dataset.data

centroids = alg.generate(data, num_clusters)

print(centroids)

est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
est.fit(dataset.data)

print(est.cluster_centers_)
