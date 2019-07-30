'''Normalized mutual information score'''

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi


def score(target, found):
    return nmi(target, found, average_method='arithmetic')

