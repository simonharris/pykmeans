'''Quick CLI utility to run an initialisation against a dataset'''

from sklearn import preprocessing
    
from datasets import loader
#from initialisations import macqueen1967 as algorithm
from initialisations import khanahmad2004 as algorithm


min_max_scaler = preprocessing.MinMaxScaler()


dataset = loader.load_iris()
data = dataset.data
#data = min_max_scaler.fit_transform(data)

K = 3

centroids = algorithm.generate(data, K, {})

print("OUTPUT:")
print(centroids)

