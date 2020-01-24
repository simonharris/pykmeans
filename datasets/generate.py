"""
Generate synthetic datasets
"""

from concurrent import futures
import itertools
import os

import numpy as np
from sklearn.datasets import make_blobs

# Ceres setup
opts_k = [2, 5, 10, 20]
opts_feats = [2, 10, 50, 100, 1000]
opts_samps = [1000]
opts_card = ['u', 'r']  # uniform or random
opts_stdev = [0.5, 1, 1.5]
N_EACH = 50
OUTPUT_DIR = './synthetic/'

"""
# Dev setup
opts_k = [20]
opts_feats = [50]
opts_samps = [1000]
opts_card = ['r']  # uniform or random
opts_stdev = [1]
N_EACH = 5
OUTPUT_DIR = './synthetic_simon/'
"""

NAME_SEP = '_'
opts_howmany = range(0, N_EACH)


def gen_dataset(no_clusters, no_feats, no_samps, card, stdev, *args):
    """Generates individual dataset"""

    if card == 'r':
        weights = np.random.random(no_clusters)
        weights /= weights.sum()
        sample_cts = np.ceil(weights * no_samps).astype(int)

        while sample_cts.sum() > no_samps:
            sample_cts[np.argmax(sample_cts)] -= 1
            # print(sample_cts.sum())

        print(sample_cts.sum())

        centers = None
    else:
        sample_cts = no_samps
        centers = no_clusters

    return make_blobs(
        n_samples=sample_cts,
        centers=centers,
        n_features=no_feats,
        cluster_std=stdev)


def gen_name(config):
    """Generates unique name for dataset"""

    config = list(config)
    index = config.pop()

    return NAME_SEP.join(map(str, config)) + NAME_SEP + f"{index:03d}"


def save_to_disk(data, labels, name):
    """Save both files to disk in their own directory"""

    dirname = OUTPUT_DIR + name + '/'

    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass  # no problem

    np.savetxt(dirname + 'data.csv', data, delimiter=',')
    np.savetxt(dirname + 'labels.csv', labels, delimiter=',')


def handler(config):
    """The callback for the executor"""

    data, labels = gen_dataset(*config)
    save_to_disk(data, labels, gen_name(config))

    print("Done with:", config)


# main code -------------------------------------------------------------------


configs = itertools.product(opts_k, opts_feats, opts_samps,
                            opts_card, opts_stdev, opts_howmany)

with futures.ProcessPoolExecutor() as executor:
    res = executor.map(handler, configs)

# Just calling this here is enough to display errors which may
# have been hidden behind the Executor's parallel operation
print(len(list(res)))
