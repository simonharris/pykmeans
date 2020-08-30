"""
Principle Component Analysis-based algorithm from Onoda 2012
"""

from sklearn.decomposition import PCA

from initialisations.onoda_base import Onoda


class OnodaPCA(Onoda):
    """Onoda 2012 PCA implementation"""

    def _find_components(self):
        """Run Pricipal Component Analysis"""

        pca = PCA(n_components=self._num_clusters)
        pca.fit(self._data)
        return pca.components_


def generate(data, num_clusters):
    """Provide consistent interface"""

    onoda = OnodaPCA(data, num_clusters)
    return onoda.find_centers()
