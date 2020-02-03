"""
Generate synthetic datasets
"""

from concurrent import futures
import itertools
import os

import numpy as np
from sklearn.datasets import make_blobs
from sklearn import preprocessing

ENV = 'dev'
# ENV = 'ceres'

if ENV == 'ceres':
    # Ceres setup
    OPTS_K = [2, 5, 10, 20]
    OPTS_FEATS = [2, 10, 50, 100, 1000]
    OPTS_SAMPS = [1000]
    OPTS_CARD = ['u', 'r']  # uniform or random
    OPTS_STDEV = [0.5, 1, 1.5]
    N_EACH = 50
    OUTPUT_DIR = './synthetic/'
    N_PROCESSES = None  # defaults to os.cpu_count()

else:
    # Dev setup
    OPTS_K = [5, 20]
    OPTS_FEATS = [50]
    OPTS_SAMPS = [1000]
    OPTS_CARD = ['r']  # uniform or random
    OPTS_STDEV = [1]
    N_EACH = 50
    OUTPUT_DIR = './synthetic/'
    N_PROCESSES = 1

NAME_SEP = '_'
MIN_CL_SIZE = 0.03
WEIGHT_SHIFT = 0.005


def _gen_dataset(no_clusters, no_feats, no_samps, card, stdev):
    """Generates individual dataset"""

    if card == 'r':
        weights = _gen_weights(no_clusters)
        sample_cts = np.ceil(weights * no_samps).astype(int)

        while sample_cts.sum() > no_samps:
            sample_cts[np.argmax(sample_cts)] -= 1

        centers = None
    else:
        sample_cts = no_samps
        centers = no_clusters

    return make_blobs(
        n_samples=sample_cts,
        centers=centers,
        n_features=no_feats,
        cluster_std=stdev)


def _gen_weights(num_clusters):
    """Generate the weightings for non-uniform clustering"""

    weights = np.random.random(num_clusters)
    weights /= weights.sum()

    # Brute-force ensure no cluster smaller than MIN_CL_SIZE
    while np.min(weights) < MIN_CL_SIZE:
        weights[np.argmin(weights)] += WEIGHT_SHIFT
        weights[np.argmax(weights)] -= WEIGHT_SHIFT

    return weights


def _gen_name(config, index):
    """Generates unique name for dataset"""

    subdir = config[0]
    subdir = f"{subdir:02d}"

    return subdir + '/' + NAME_SEP.join(map(str, config)) + \
        NAME_SEP + f"{index:03d}"


def _save_to_disk(data, labels, config, index):
    """Save both files to disk in their own directory"""

    name = _gen_name(config, index)

    dirname = OUTPUT_DIR + name + '/'

    # Standardise data
    data = preprocessing.scale(data)

    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass  # no problem

    np.savetxt(dirname + 'data.csv', data, delimiter=',')
    np.savetxt(dirname + 'labels.csv', labels, fmt="%i", delimiter=',')


def _handler(config):
    """The callback for the executor"""

    config = list(config)
    index = config.pop()

    data, labels = _gen_dataset(*config)
    _save_to_disk(data, labels, config, index)

    print("Done with:", config)


def main():
    """Main method"""
    configs = itertools.product(OPTS_K, OPTS_FEATS, OPTS_SAMPS,
                                OPTS_CARD, OPTS_STDEV, range(0, N_EACH))

    with futures.ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:
        res = executor.map(_handler, configs)

    # Just calling this here is enough to display errors which may
    # have been hidden behind the Executor's parallel operation
    print(len(list(res)))


# Main ------------------------------------------------------------------------


if __name__ == '__main__':
    main()
