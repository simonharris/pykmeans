
"""Temp bootstrap file to run K&A"""
import pathhack

from datasets import testloader
from initialisations import yuan2004 as alg
from preprocessors import stddise

from sklearn.cluster import KMeans

# dataset = testloader.load_fossil()
# dataset = testloader.load_iris()

# Didn't complete on Ceres
dataset = testloader._load_local('2_2_1000_u_1_048')
data = dataset.data

num_clusters = 2

centroids = alg.generate(data, num_clusters)

print(centroids)
"""
est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
est.fit(dataset.data)

print(est.cluster_centers_)
"""
