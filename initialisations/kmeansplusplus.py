import numpy as np
import kmeans

'''
Arthur & Vassilvitskii k-means++ algorithm

See: k-means++: The advantages of careful seeding
http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf
'''

def generate(data, K):

    # Initial centroid
    randindex = np.random.choice(data.shape[0], replace=False)
    centroids = np.array([data[randindex]])

    # Remaining required centroids
    while len(centroids) < K:

        distances = kmeans.distance_table(data, centroids)
        probabilities = distances.min(1)**2 / np.sum(distances.min(1)**2)

        randindex = np.random.choice(data.shape[0], replace=False, p=probabilities)
        centroids = np.append(centroids, [data[randindex]], 0)

    return centroids