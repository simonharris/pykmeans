import numpy as np
from sklearn.cluster import KMeans

'''
Likas 1993 Global K-Means algorithm

See: The global k-means clustering algorithm
https://www.sciencedirect.com/science/article/pii/S0031320302000602
'''

def run_k_means(data, centroids):
    
    best = None
    
    for row in data:
                
        candidates = np.concatenate((centroids, [row]), axis=0)
    
        estimator = KMeans(n_clusters=len(candidates), init=np.array(candidates), n_init=1)
        estimator.fit(data)
        
        if best == None or estimator.inertia_ < best.inertia_:
            best = estimator
 
    return best.cluster_centers_
    
    
def generate(data, K):

    centroids = [np.mean(data, axis=0)]

    while len(centroids) < K:
        centroids = run_k_means(data, centroids)
    
    return centroids
    
