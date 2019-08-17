

from datasets import loader
from initialisations import macqueen1967 as algorithm

dataset = loader.load_iris()
#dataset = loader.load_hartigan()



K = 3

centroids = algorithm.generate(dataset.data, K, {})

print("OUTPUT:")
print(centroids)


