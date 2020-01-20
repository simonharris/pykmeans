"""
Generate synthetic datasets
"""

from concurrent import futures
import itertools
import os

import numpy as np
from sklearn.datasets import make_blobs


opts_k = [2, 5, 10, 20]
opts_feats = [2, 10, 50, 100, 1000]
opts_samps = [1000]
opts_card = ['u', 'r']  # uniform or random
opts_stdev = [0.5, 1, 1.5]

N_EACH = 50

opts_index = range(0, N_EACH)

NAME_SEP = '_'
OUTPUT_DIR = './synthetic/'


def gen_dataset(no_clusters, no_feats, no_samps, card, stdev, *args):
    """Generates individual dataset"""

    if card == 'r':
        weights = np.random.random(no_clusters)
        weights /= weights.sum()
        samples = (weights * no_samps).astype('int')
        print(samples.sum())
        centers = None
    else:
        samples = no_samps
        centers = no_clusters

    return make_blobs(
        n_samples=samples,
        n_features=no_feats,
        centers=centers,
        cluster_std=stdev)


def gen_name(config):
    """Generates unique name for dataset"""

    return NAME_SEP.join(map(str, config))


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
    save_to_disk(data, labels, gen_name(config))  # , i))

    print("Done with:", config)


# main code -------------------------------------------------------------------


configs = itertools.product(opts_k, opts_feats, opts_samps,
                            opts_card, opts_stdev, opts_index)

with futures.ProcessPoolExecutor() as executor:
    res = executor.map(handler, configs)

# print(len(list(res)))
