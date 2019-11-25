"""Initial test runner so we can get some idea of timings"""

import csv
import importlib
import itertools
import os
import time

import numpy as np
from sklearn.cluster import KMeans

from dataset import Dataset
from metrics import ari

DIR_REAL = 'datasets/realworld/'


def load_dataset(which):
    """Load a dataset from disk"""

    datafile = DIR_REAL + which + '/data.csv'
    labelfile = DIR_REAL + which + '/labels.csv'

    return Dataset(
        which,
        np.loadtxt(datafile, delimiter=',', dtype=np.int),
        np.loadtxt(labelfile, delimiter=',', dtype=np.int),
    )


def run_kmeans(dataset, algorithm):
    """Run the initialisation algorithm follower by k-means"""

    num_clusters = dataset.num_clusters()

    centroids = algorithm.generate(dataset.data, num_clusters)

    est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
    est.fit(dataset.data)

    return est.labels_, est.inertia_


def save_log_file(info):
    """Write info to disk"""

    outfile = '_output/output-' + str(time.time()) + '.csv'
    with open(outfile, 'w+') as my_csv:
        csvwriter = csv.writer(my_csv, delimiter=',')
        csvwriter.writerows(info)

# -----------------------------------------------------------------------------


algorithms = ['faber1994'] # , 'kmeansplusplus']  # , 'erisoglu2011']

datasets = [d for d in os.listdir(DIR_REAL)
            if os.path.isdir(os.path.join(DIR_REAL, d))]

configs = itertools.product(datasets, algorithms)

output = []

for dsname, algname in configs:

    log = []
    log.append(algname)
    log.append(dsname)

    print(log)

    my_init = importlib.import_module('initialisations.' + algname)

    dset = load_dataset(dsname)

    try:
        labels, inertia = run_kmeans(dset, my_init)

        log.append(inertia)

        ari = ari.score(dset.target, labels)
        log.append(ari)


#        # add time

        print(log)
        output.append(log)

    except:
        continue


save_log_file(log)
