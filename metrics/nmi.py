from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

'''Normalized mutual information score'''


def score(target, found):
    return nmi(target, found, average_method='arithmetic')

