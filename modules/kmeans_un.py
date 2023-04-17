import numpy as np

class kmeans():
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def fit(self, X):
         # Inicializar centroides aleatorios
        self.centroids = X[np.random.choice(X.shape[0], self.K, replace=False)]
        for i in range(self.max_iters):
            # Calcular la distancia entre los datos y los centroides
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # Asignar cada punto al cluster más cercano
            self.labels = np.argmin(distances, axis=0)
            # Actualizar los centroides
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.K)])
            # Verificar si los centroides han cambiado
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids, self.labels

    def predict(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
    