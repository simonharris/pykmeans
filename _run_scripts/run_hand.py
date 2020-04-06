"""Temp bootstrap file to run Hand 2005"""
import pathhack

from datasets import testloader
from initialisations import hand2005 as alg

dataset = testloader._load_local('20_1000_1000_u_1_005')
num_clusters = 20

centroids = alg.generate(dataset.data, num_clusters)

print(centroids)
