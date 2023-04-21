import numpy as np

class KMedoids:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X):
        # Randomly initialize medoids
        #rng = np.random.RandomState(self.random_state)
        rng = np.random.RandomState(2)
        self.medoids = rng.choice(X.shape[0], self.n_clusters, replace=False)
        
        for i in range(self.max_iter):
            # Assign each point to the nearest medoid.
            distances = np.abs(X[:, np.newaxis] - X[self.medoids])
            cluster_labels = np.argmin(np.sum(distances, axis=2), axis=1)
            
            # Updating the medoids
            for j in range(self.n_clusters):
                mask = cluster_labels == j
                cluster_points = X[mask]
                cluster_distances = np.sum(np.abs(cluster_points[:, np.newaxis] - cluster_points), axis=2)
                costs = np.sum(cluster_distances, axis=1)
                best_medoid_idx = np.argmin(costs)
                self.medoids[j] = np.where(mask)[0][best_medoid_idx]
        
        self.cluster_labels_ = cluster_labels
        self.medoids_ = [X[i,:] for i in self.medoids]
        return self.medoids_, self.cluster_labels_
    
    def predict(self, X):
        distances = np.abs(X[:, np.newaxis] - X[self.medoids])
        cluster_labels = np.argmin(np.sum(distances, axis=2), axis=1)
        return cluster_labels
