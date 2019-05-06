import numpy as np

def load_spambase():
    '''Load the spambase dataset.
    See: https://archive.ics.uci.edu/ml/datasets/Spambase'''

    file = '../sample_data/spambase/spambase.data'

    dataset  = np.loadtxt(file, delimiter=',', dtype=np.double)

    data = dataset[:,:-1]
    target = dataset[:,-1]

    return {'data':data, 'target':target}
