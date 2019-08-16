'''Mainly to remove duplicated code from many notebooks'''

from sklearn.cluster import KMeans

from datasets import loader
from metrics import accuracy, nmi, ari


SEP = '====================================================='


def run_clustering(algorithm, opts):
    '''Run all datasets'''

    _run_dataset('HART', loader.load_hartigan(), 3, algorithm, opts)
    _run_dataset('IRIS', loader.load_iris(), 3, algorithm, opts)
    _run_dataset('SOYS', loader.load_soy_small(), 4, algorithm, opts)
    _run_dataset('WINE', loader.load_wine(), 3, algorithm, opts)
    _run_dataset('WBCO', loader.load_wbco(), 2, algorithm, opts)


def _run_dataset(name, dataset, K, algorithm, opts):
    '''Run individual dataset'''

    print("Running " + name)
    print(SEP)
    
    data = dataset.data
    target = dataset.target

    centroids = algorithm.generate(data, K, opts)

    #print("Initial seeds:")
    #print(centroids, "\n")

    est = KMeans(n_clusters=K, n_init=1, init=centroids)
    est.fit(data)
    
    #print("Discovered centroids:")
    #print(est.cluster_centers_, "\n")
    
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
    
