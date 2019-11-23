"""
Independent Component Analysis-based algorithm from Onoda 2012
"""

from sklearn.decomposition import FastICA

from initialisations.onoda2012 import Onoda


class OnodaICA(Onoda):

    @staticmethod
    def _find_components(data, num_components):
        """Run Independent Component Analysis"""

        ica = FastICA(n_components=num_components)
        ica.fit_transform(data)
        return ica.components_


def generate(data, num_clusters):
    """Provide consistent interface"""

    onoda = OnodaICA(data, num_clusters)
    return onoda.find_centers(data, num_clusters)
