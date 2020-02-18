"""Temp bootstrap file to run K&A"""
import pathhack

from datasets import testloader
from initialisations import erisoglu2011 as alg
from preprocessors import stddise

# dataset = testloader.load_fossil()
# dataset = testloader.load_iris()

# Didn't complete on Ceres
dataset = testloader._load_local('2_1000_1000_r_0.5_009')

num_clusters = 20

data = stddise.process(dataset.data)

centroids = alg.generate(data, num_clusters)

print(centroids)
