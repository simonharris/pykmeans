"""Mainly to remove duplicated code from many notebooks"""

from sklearn.cluster import KMeans

from datasets import testloader
from metrics import accuracy, nmi, ari


SEP = '====================================================='


def run_clustering(algorithm):
    """Run all datasets"""

    _run_dataset('HART', testloader.load_hartigan(), 3, algorithm)
    _run_dataset('IRIS', testloader.load_iris(), 3, algorithm)
    _run_dataset('SOYS', testloader.load_soy_small(), 4, algorithm)
    _run_dataset('WINE', testloader.load_wine(), 3, algorithm)
    _run_dataset('WBCO', testloader.load_wbco(), 2, algorithm)


def _run_dataset(name, dataset, num_clusters, algorithm):
    """Run individual dataset"""

    print("Running " + name)
    print(SEP)

    data = dataset.data
    target = dataset.target

    try:
        centroids = algorithm.generate(data, num_clusters)
    except BaseException as myerror:
        print("Exception:", myerror, "\n")
        return

    # print("Initial seeds:")
    # print(centroids, "\n")

    est = KMeans(n_clusters=num_clusters, n_init=1, init=centroids)
    est.fit(data)

    # print("Discovered centroids:")
    # print(est.cluster_centers_, "\n")

    _run_metrics(target, est)



def _run_metrics(target, est):
    '''Run all the evaluation metrics'''

    acc = accuracy.score(target, est.labels_)
    asc = ari.score(target, est.labels_)
    nsc = nmi.score(target, est.labels_)

    print("Accuracy Score:", acc)
    print("Adjusted Rand Index:", asc)
    print("NMI:", nsc)
    print("Inertia:", est.inertia_, "\n")


def run_kmeans_verbose(dataset, seeds):
    """Run kmeans and print detailed results"""

    est = KMeans(n_clusters=len(seeds), n_init=1, init=seeds)
    est.fit(dataset.data)

    print("Discovered centroids:")
    print(est.cluster_centers_, "\n")
    print("Labels:")
    print(est.labels_, "\n")

    _run_metrics(dataset.target, est)
