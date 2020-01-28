"""
Initial skeleton for a paralellized runner
"""

from concurrent import futures
import importlib
import itertools
import os
import time

import numpy as np

from dataset import Dataset
from metrics import ari


# Run-specific config
WHICH_SETS = 'synthetic'
algorithms = ['random']
N_RUNS = 1  # 50


DATASETS = './datasets/' + WHICH_SETS + '/'
DIR_OUTPUT = '_output/'


def find_datasets(directory):
    """Find all datasets in a given directory"""

    return [d for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d))]


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


def handler(config):
    """Main processing method"""

    algname = config[0]
    datadir = config[1]
    ctr = config[2]

    start = time.perf_counter()

    log = []
    log.append(algname)
    log.append(datadir)
    print(log)

    my_output_dir = DIR_OUTPUT + algname + '/' + WHICH_SETS + '/' + datadir + '/'
    os.makedirs(my_output_dir)

    # print("Called with:", my_output_dir)

    dset = load_dataset(DATASETS, datadir)
    my_init = importlib.import_module('initialisations.' + algname)
    labels, inertia = run_kmeans(dset, my_init)

    log.append(inertia)

    print(labels)
    ### TODO: save label file

    ari_score = ari.score(dset.target, labels)
    log.append(ari_score)

    print(log)

    # add time taken
    end = time.perf_counter()
    log.append(end - start)


    ### TODO: write log fle

    # print("OK SO FAR")


all_sets = find_datasets(DATASETS)

configs = itertools.product(algorithms, all_sets, range(0, N_RUNS))

with futures.ProcessPoolExecutor() as executor:
    res = executor.map(handler, configs)

# Force errors to be printed
print(len(list(res)))
