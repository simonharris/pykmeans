"""Quick CLI utility to run an initialisation against a dataset"""

from datasets import loader
from initialisations import khanahmad2004 as algorithm

dataset = loader.load_iris_ccia()
data = dataset.data

K = 3

centroids = algorithm.generate(data, K, {})

print("OUTPUT:")
print(centroids)
