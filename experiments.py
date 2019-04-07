import sklearn.datasets as skdatasets
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics
import kmeans
import utils
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
    #'ce705': utils.get_learning_data
}

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", dest="dataset",
                    help="Which dataset to use", default="iris")
parser.add_argument("-k", "--K", dest="K",
                    help="K", default=3)

args = parser.parse_args()

print("Running experiments with configuration:", args, "\n")

try:
    dataloader = datasets[args.dataset]
    dataset = dataloader()
    K = int(args.K)
except:
    print("Helpful error message")
    sys.exit()

data = utils.standardise(dataset.data)
target = dataset.target

Z, U, clusters, iterations = kmeans.cluster(data, K)

est1 = skcluster.KMeans(n_clusters=K, n_init=1, init='random')
est1.fit(data)

est2 = skcluster.KMeans(n_clusters=K)
est2.fit(data)

print("")
print('Me:\n', U)
print("SKL (naive):\n", est1.labels_)
print("SKL (smarter):\n", est2.labels_)
print("Target:\n", target)

score_me = skmetrics.adjusted_rand_score(target, U)
score_them_n = skmetrics.adjusted_rand_score(target, est1.labels_)
score_them_s = skmetrics.adjusted_rand_score(target, est2.labels_)

print("\n----------------------------------------------------------------")
print("\nAdjusted Rand Index:")
print("Me:", score_me, "| SKL (naive):", score_them_n, "| SKL (smarter):", score_them_s)
print("")
