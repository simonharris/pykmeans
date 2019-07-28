import sklearn.metrics as skmetrics
from metrics import accuracy, nmi

'''Mainly to remove duplicated code from many notebooks'''


def run_metrics(target, est):
    '''Run all the evaluation metrics'''

    acc = accuracy.score(target, est.labels_)
    ari = skmetrics.adjusted_rand_score(target, est.labels_)
    nsc = nmi.score(target, est.labels_)

    print("Accuracy Score:", acc)
    print("Adjusted Rand Index:", ari)
    print("NMI:", nsc)
    print("Inertia:", est.inertia_)
    
