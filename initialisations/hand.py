"""Hand & Krzanowski 2005 algorithm"""

import numpy as np

from initialisations import random as randominit
from initialisations.base import EmptyClusterException, Initialisation


class Hand(Initialisation):
    """Hand & Krzanowski 2005 initialisation algorithm"""

    # As per the paper
    _iterations = 100
    _alpha = 0.3
    _beta = 0.95
    _converge_limit = 10

    def find_centers(self):
        """Main method"""

        converge_count = 0

        # Step 1: initial run of k-means from random start points
        start_point = randominit.generate(self._data, self._num_clusters)
        labels, centers, inertia = self._run_k_means(start_point)

        alpha = self._alpha
        last_inertia = inertia

        for _ in range(0, self._iterations):

            # Step 2: "perturbate the configuration"
            p_labels = self._perturbate(labels, alpha)

            try:
                p_centers = self._find_new_centers(p_labels)
            except EmptyClusterException:
                # print("!!! Exception caught")
                continue

            new_labels, new_centers, new_inertia = self._run_k_means(p_centers)
            # print("New:", new_inertia, "; Old:", inertia)

            if new_inertia < inertia:
                # print("\t==> Keeping new configuration")
                inertia = new_inertia
                labels = new_labels
                centers = new_centers

            if new_inertia == last_inertia:
                # print("\t==> Inertia same as last time")
                converge_count += 1
            else:
                converge_count = 0

            last_inertia = new_inertia

            if converge_count >= self._converge_limit:
                # print("\t==> Breaking at:", converge_count)
                break

            # Step 3: reduce probability of perturbation
            alpha = alpha * self._beta

        return centers

    def _perturbate(self, my_labels, alpha):
        """Randomly reassign data points with probability alpha"""

        all_labels = range(0, self._num_clusters)

        probabilities = np.random.random_sample(self._num_samples)

        for selected in np.where(probabilities < alpha)[0]:
            my_labels[selected] = np.random.choice(
                np.where(all_labels != my_labels[selected])[0])

        return my_labels

    def _find_new_centers(self, my_new_labels):
        """Find the means of the perturbed clusters"""

        new_centers = np.zeros((self._num_clusters, self._num_attrs))

        for cluster_id in range(0, self._num_clusters):
            cluster_data = self._data[my_new_labels == cluster_id]

            if cluster_data.size == 0:
                raise EmptyClusterException("Empty cluster:" + str(cluster_id))

            new_centers[cluster_id] = np.mean(cluster_data, axis=0)

        return new_centers


def generate(data, num_clusters):
    """The common interface"""

    hand = Hand(data, num_clusters)
    return hand.find_centers()
