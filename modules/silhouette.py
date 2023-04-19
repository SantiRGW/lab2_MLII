import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def silhouette_coefficient(data, labels, point_index):
    # Obtener el label del cluster del punto
    point_label = labels[point_index]
    # Obtener las distancias euclidianas del punto a los demás puntos en su cluster y a los puntos en los otros clusters
    intra_cluster_distances = [euclidean_distance(data[point_index], data[i]) for i in range(len(data)) if labels[i] == point_label and i != point_index]
    inter_cluster_distances = [np.mean([euclidean_distance(data[point_index], data[i]) for i in range(len(data)) if labels[i] == j]) for j in np.unique(labels) if j != point_label]
    # Calcular el coeficiente de silueta para el punto
    if len(intra_cluster_distances) > 0 and len(inter_cluster_distances) > 0:
        a = np.mean(intra_cluster_distances)
        b = np.min(inter_cluster_distances)
        return (b - a) / np.max([a, b])
    else:
        return 0
    
def silhouette_score(data, labels):
    # Calcular el coeficiente de silueta para cada punto en los datos
    silhouette_scores = [silhouette_coefficient(data, labels, i) for i in range(len(data))]
    # Calcular el promedio del coeficiente de silueta para todos los puntos
    #return np.mean(silhouette_scores)
    return silhouette_scores,np.mean(silhouette_scores)

