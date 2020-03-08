"""Temp bootstrap file to run K&A"""

from datasets import testloader
from initialisations import khanahmad2004 as ka

dataset = testloader.load_fossil()
# dataset = testloader.load_iris()

# Fossil and Iris both have 3
num_clusters = 3

centroids = ka.generate(dataset.data, num_clusters)

#print(centroids)
