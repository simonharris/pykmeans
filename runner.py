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

# Don't nag me about constants
# pylint: disable=C0103


DIR_REAL = 'datasets/realworld/'
DIR_SYNTH = 'datasets/synthetic/'

MY_DIR = DIR_REAL


def find_datasets(directory):
    """Find all datasets in a given directory"""

    return [d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(DIR_REAL, d))]


def load_dataset(datadir, which):
    """Load a dataset from disk"""

    datafile = datadir + which + '/data.csv'
    labelfile = datadir + which + '/labels.csv'

    return Dataset(
        which,
        np.loadtxt(datafile, delimiter=','),
        np.loadtxt(labelfile, delimiter=','),
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
    print("Saving to:", outfile)
    with open(outfile, 'w+') as my_csv:
        csvwriter = csv.writer(my_csv, delimiter=',')
        csvwriter.writerows(info)

# -----------------------------------------------------------------------------


algorithms = [
        # 'erisoglu2011',
        'faber1994',
        'kkz1994',
        'kmeansplusplus',
        # 'onoda2012ica',
        ]
datasets = find_datasets(MY_DIR)
configs = itertools.product(datasets, algorithms)

output = []

for dsname, algname in configs:

    log = []
    log.append(algname)
    log.append(dsname)
    print(log)

    start = time.perf_counter()

    my_init = importlib.import_module('initialisations.' + algname)

    dset = load_dataset(MY_DIR, dsname)

    labels, inertia = run_kmeans(dset, my_init)
    log.append(inertia)

    ari_score = ari.score(dset.target, labels)
    log.append(ari_score)

    # add time
    end = time.perf_counter()
    log.append(end - start)

    print(log)
    output.append(log)

save_log_file(output)
