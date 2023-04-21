import numpy as np

class kmeans():
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def fit(self, X):
         # Initialize random centroids
        self.centroids = X[np.random.choice(X.shape[0], self.K, replace=False)]
        for i in range(self.max_iters):
            # Calculate distance between data and centroids
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # Assign each point to the nearest cluster
            self.labels = np.argmin(distances, axis=0)
            # Update the centroids
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.K)])
            # Check if the centroids have changed
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids, self.labels

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
    