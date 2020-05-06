"""Temp bootstrap file to run K&A"""
import pathhack

from datasets import testloader
from initialisations import erisoglu2011 as alg

from sklearn.cluster import KMeans

"""
./avilatr/exceptions-000.csv
./ecoli/exceptions-000.csv
./glass/exceptions-000.csv
./letterrec/exceptions-000.csv
./optdigits/exceptions-000.csv
./pendigits/exceptions-000.csv
./wineq_red/exceptions-000.csv
"""


# Didn't initially complete on Ceres
dataset = testloader._load_local('20_2_1000_r_1.5_035')
num_clusters = 20

# Exceptions on Ceres
# dataset = testloader._load_local('wineq_red')
# num_clusters = 6

data = dataset.data

centroids = alg.generate(data, num_clusters)

print(centroids)

est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
est.fit(dataset.data)

print(est.cluster_centers_)
