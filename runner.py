'''Quick CLI utility to run an initialisation against a dataset'''

from datasets import loader
#from initialisations import macqueen1967 as algorithm
from initialisations import khanahmad2004 as algorithm


dataset = loader.load_iris()

K = 3

centroids = algorithm.generate(dataset.data, K, {})

print("OUTPUT:")
print(centroids)

