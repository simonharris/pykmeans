"""Temp bootstrap file to run Pavan et al. SPSS"""
import pathhack

from datasets import testloader
from initialisations import spss2010 as alg

from sklearn.cluster import KMeans

dataset = testloader.load_iris()
num_clusters = 3

# dataset = testloader.load_hartigan()
# num_clusters = 3

# dataset = testloader.load_soy_small()
# num_clusters = 4

data = dataset.data
centroids = alg.generate(data, num_clusters)

#print(centroids)
"""
est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
est.fit(dataset.data)

print(est.cluster_centers_)
"""
