"""Temp bootstrap file to run ikmeans"""
import pathhack

from datasets import testloader
from initialisations import ikmeans_card as alg
from preprocessors import stddise

# from sklearn.cluster import KMeans

# Seems to hang on Ceres
dataset = testloader._load_local('wbco')
num_clusters = 2

data = dataset.data
centroids = alg.generate(data, num_clusters)

print(centroids)
"""
est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
est.fit(dataset.data)

print(est.cluster_centers_)
"""
