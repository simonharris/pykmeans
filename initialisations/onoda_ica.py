"""
Independent Component Analysis-based algorithm from Onoda 2012
"""

from sklearn.decomposition import FastICA

from initialisations.onoda_base import Onoda


class OnodaICA(Onoda):
    """Onoda 2012 ICA implementation"""

    def _find_components(self):
        """Run Independent Component Analysis"""

        ica = FastICA(n_components=self._num_clusters, max_iter=1000)
        ica.fit_transform(self._data)
        return ica.components_


def generate(data, num_clusters):
    """Provide consistent interface"""

    onoda = OnodaICA(data, num_clusters)
    return onoda.find_centers()
