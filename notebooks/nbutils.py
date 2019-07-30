'''Mainly to remove duplicated code from many notebooks'''

from metrics import accuracy, nmi, ari


def run_metrics(target, est):
    '''Run all the evaluation metrics'''

    acc = accuracy.score(target, est.labels_)
    asc = ari.score(target, est.labels_)
    nsc = nmi.score(target, est.labels_)

    print("Accuracy Score:", acc)
    print("Adjusted Rand Index:", asc)
    print("NMI:", nsc)
    print("Inertia:", est.inertia_)
    
