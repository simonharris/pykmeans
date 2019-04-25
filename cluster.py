from scipy.spatial import distance as spdistance

class Cluster():

    def __init__(self, centroid, data):
        self._centroid = centroid
        self._data = data

    def sum_distances(self):
        return sum([spdistance.euclidean(self._centroid, point) for point in self._data])
