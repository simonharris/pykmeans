"""
Independent Component Analysis-based algorithm from Onoda 2012
"""

from sklearn.decomposition import FastICA

from initialisations.onoda2012 import Onoda


class OnodaICA(Onoda):
    """Onoda 2012 ICA implementation"""

    @staticmethod
    def _find_components(data, num_clusters):
        """Run Independent Component Analysis"""

        ica = FastICA(n_components=num_clusters)
        ica.fit_transform(data)
        return ica.components_


def generate(data, num_clusters):
    """Provide consistent interface"""

    onoda = OnodaICA(data, num_clusters)
    return onoda.find_centers(data, num_clusters)
