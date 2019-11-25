"""Dataset class definition"""

import numpy as np


class Dataset():
    """Mimics the sklearn dataset interface"""

    def __init__(self, name, data, target):
        self.name = name
        self.data = data
        self.target = target

    def num_clusters(self):
        """Calculate the number of clusters"""

        return len(np.unique(self.target))
