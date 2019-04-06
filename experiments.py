from sklearn import datasets
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import kmeans
import utils

'''
See:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
'''

dataset = datasets.load_iris()
#wine = datasets.load_wine()
#bc = datasets.load_breast_cancer()

K = 3

data = utils.standardise(dataset.data)
target = dataset.target

Z, U, clusters, iterations = kmeans.cluster(data, K)

est1 = cluster.KMeans(n_clusters=K, n_init=1, init='random')
est1.fit(data)

est2 = cluster.KMeans(n_clusters=K)
est2.fit(data)

print('Mine:\n', U)
print("SKL (naive):\n", est1.labels_)
print("SKL (smarter):\n", est2.labels_)
print("Target:\n", target)

score_me = metrics.adjusted_rand_score(target, U)
score_them_n = metrics.adjusted_rand_score(target, est1.labels_)
score_them_s = metrics.adjusted_rand_score(target, est2.labels_)

print("----------------------------------------------------------------")
print("Me:", score_me, "| SKLearn (naive):", score_them_n, "| SKLearn (smarter):", score_them_s)
