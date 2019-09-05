"""
KKZ 1994 algorithm

See: A new initialization technique for generalized Lloyd iteration
https://ieeexplore.ieee.org/abstract/document/329844/
"""

import numpy as np

from kmeans import distance_table
from initialisations.base import Initialisation


class KKZ(Initialisation):
    """KKZ 1994 initialisation algorithm"""

    def find_centers(self):
        """Main method"""

        norms = np.linalg.norm(self._data, axis=1)

        first = self._data[np.argmax(norms)]
        codebook = np.array([first])

        while codebook.shape[0] < self._num_clusters:
            distances = distance_table(self._data, codebook)
            mins = np.min(distances, axis=1)
            amax = np.argmax(mins, axis=0)
            nxt = self._data[amax]
            codebook = np.append(codebook, [nxt], axis=0)

        return codebook


# -----------------------------------------------------------------------------


def generate(data, num_clusters, opts):
    """The common interface"""

    kkz = KKZ(data, num_clusters, opts)
    return kkz.find_centers()
