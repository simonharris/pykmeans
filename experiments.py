import sklearn.datasets as skdatasets
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics
import kmeans
import utils
from initialisations import random
import sys
from argparse import ArgumentParser

'''
See:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
'''

datasets = {
    'iris':  skdatasets.load_iris,
    'wine':  skdatasets.load_wine,
    'bc':    skdatasets.load_breast_cancer,
}

algorithms = {
    'random': random.generate,
    #ikmeans
    #erisolgiu
}

# functions --------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset",
                        help="Which dataset to use", default="iris")
    parser.add_argument("-k", "--K", dest="K",
                        help="K", default=3)
    parser.add_argument("-a", "--algorithm", dest="algorithm",
                        help="Which initialisation algorithm to use", default='random')

    return parser.parse_args()

# Run main script --------------------------------------------------------------

args = parse_args()
print("Running experiments with configuration:", args, "\n")

try:
    dataloader = datasets[args.dataset]
    initialiser = algorithms[args.algorithm]
except:
    print("Helpful error message to follow")
    sys.exit()

dataset = dataloader()
K = int(args.K)

data = utils.standardise(dataset.data)
target = dataset.target

# Run initialisation algorithm
centroids = initialiser(data, K)
print("Centroids:\n", centroids)

# Run clustering algorithm
Z, U, clusters, iterations = kmeans.cluster(data, K, centroids)

est1 = skcluster.KMeans(n_clusters=K, n_init=1, init='random')
est1.fit(data)

est2 = skcluster.KMeans(n_clusters=K)
est2.fit(data)

print("")
print('Me:\n', U)
print("SKL (naive):\n", est1.labels_)
print("SKL (smarter):\n", est2.labels_)
print("Target:\n", target)

# Run metrics ------------------------------------------------------------------

print("\n----------------------------------------------------------------")
print("Metrics:")

acc_me = skmetrics.accuracy_score(target, U)
acc_them_n = skmetrics.accuracy_score(target, est1.labels_)
acc_them_s = skmetrics.accuracy_score(target, est2.labels_)

print("\nAccuracy Score:")
print("Me:", acc_me, "| SKL (naive):", acc_them_n, "| SKL (smarter):", acc_them_s)

ari_me = skmetrics.adjusted_rand_score(target, U)
ari_them_n = skmetrics.adjusted_rand_score(target, est1.labels_)
ari_them_s = skmetrics.adjusted_rand_score(target, est2.labels_)

print("\nAdjusted Rand Index:")
print("Me:", ari_me, "| SKL (naive):", ari_them_n, "| SKL (smarter):", ari_them_s)
print("")
