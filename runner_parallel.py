"""
Initial skeleton for a paralellized runner
"""

from concurrent import futures
import csv
import importlib
import itertools
import os
import time
# import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from dataset import Dataset
from metrics import ari


# Run-specific config
WHICH_SETS = 'synthetic'
algorithms = ['random']
N_RUNS = 50


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


def run_kmeans(dataset, algorithm, dsname, ctrstr):
    """Run the initialisation algorithm follower by k-means"""

    num_clusters = dataset.num_clusters()

    centroids = algorithm.generate(dataset.data, num_clusters)

    # TODO: this currently catches nothing
    try:
        est = KMeans(n_clusters=num_clusters, init=centroids, n_init=1)
        est.fit(dataset.data)
    except ConvergenceWarning as con_warn:
        print("ConvergenceWarning for", dsname, "at:", ctrstr)
        print(con_warn)

    return est.labels_, est.inertia_


def make_output_dir(output_dir):
    """Create empty directory for output if needed"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def save_log_file(outdir, info, ctrstr):
    """Write info to disk"""

    info.append(ctrstr)
    infofile = outdir + 'output-' + ctrstr + '.csv'

    print('Saving info to:', infofile)
    with open(infofile, 'w+') as my_csv:
        csvwriter = csv.writer(my_csv, delimiter=',')
        csvwriter.writerows([info])


def save_label_file(outdir, labels, ctrstr):
    """Write discovered clustering to disk"""

    labelfile = outdir + 'labels-' + ctrstr + '.csv'

    print('Saving labels to:', labelfile)
    with open(labelfile, 'w+') as my_csv:
        csvwriter = csv.writer(my_csv, delimiter=',')
        csvwriter.writerow(labels)


def handler(config):
    """Main processing method"""

    algname = config[0]
    datadir = config[1]
    ctr = config[2]
    ctr_str = f"{ctr:03d}"

    start = time.perf_counter()

    log = []
    log.append(algname)
    log.append(datadir)
    print(log)

    my_out_dir = DIR_OUTPUT + algname + '/' + WHICH_SETS + '/' + datadir + '/'
    make_output_dir(my_out_dir)

    # print("Called with:", my_output_dir)

    dset = load_dataset(DATASETS, datadir)
    my_init = importlib.import_module('initialisations.' + algname)
    labels, inertia = run_kmeans(dset, my_init, datadir, ctr_str)

    log.append(inertia)

    # print(labels)

    ari_score = ari.score(dset.target, labels)
    log.append(ari_score)

    # add time taken
    end = time.perf_counter()
    log.append(end - start)

    print(log)

    save_log_file(my_out_dir, log, ctr_str)
    save_label_file(my_out_dir, labels, ctr_str)


# Main loop -------------------------------------------------------------------


if __name__ == '__main__':

    all_sets = find_datasets(DATASETS)

    configs = itertools.product(algorithms, all_sets, range(0, N_RUNS))

    with futures.ProcessPoolExecutor() as executor:
        res = executor.map(handler, configs)

    # Force errors to be printed
    print(len(list(res)))
