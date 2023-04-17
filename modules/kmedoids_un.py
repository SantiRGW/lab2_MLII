import numpy as np

class KMedoids:
    def __init__(self, K, max_iters=100):
        self.K = K
        self.max_iters = max_iters
    
    def fit(self, X):
        # Inicializar medoids aleatorios
        self.medoids = X[np.random.choice(X.shape[0], self.K, replace=False)]
        for i in range(self.max_iters):
            # Calcular la distancia entre los datos y los medoids
            distances = np.sqrt(((X - self.medoids[:, np.newaxis])**2).sum(axis=2))
            # Asignar cada punto al medoid más cercano
            self.labels = np.argmin(distances, axis=0)
            # Calcular el costo total
            cost = np.sum(np.min(distances, axis=0))
            # Intentar actualizar los medoids
            for k in range(self.K):
                # Obtener los puntos asignados a este medoid
                mask = (self.labels == k)
                if np.sum(mask) == 0:
                    # Si no hay puntos asignados a este medoid, saltar a la siguiente iteración
                    continue
                # Calcular el costo si se cambia el medoid a cada punto asignado
                indices = np.where(mask)[0]
                new_medoids = X[indices]
                new_distances = np.sqrt(((X - new_medoids[:, np.newaxis])**2).sum(axis=2))
                new_cost = np.sum(np.min(new_distances, axis=0))
                # Actualizar el medoid si se reduce el costo
                if new_cost < cost:
                    self.medoids[k] = X[indices[np.argmin(new_distances, axis=0)]]
                    cost = new_cost
                else:
                    # Si no se puede reducir el costo, mantener el medoid actual
                    self.medoids[k] = self.medoids[k]
    
    def predict(self, X):
        distances = np.sqrt(((X - self.medoids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
