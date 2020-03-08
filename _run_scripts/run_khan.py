"""Temp bootstrap file to run Khan 2012"""

from datasets import testloader
from initialisations import khan2012 as alg

# dataset = testloader.load_fossil()
dataset = testloader.load_iris()

# Fossil and Iris both have 3
num_clusters = 3

centroids = alg.generate(dataset.data, num_clusters)

print(centroids)
