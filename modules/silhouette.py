import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import cv2

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

def silhouette_plot_kmeans(x_list,x_labels,n_plots):
    fig = plt.figure(figsize=(13, 5))
    for j in range(n_plots):
        x=x_list[j]
        labels=x_labels[j]
        silhouette_vals = silhouette_samples(x, labels)
        silhouette_avg = np.mean(silhouette_vals)
        cluster_labels = np.unique(labels)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = [silhouette_vals[labels == i] for i in cluster_labels]
        ax = fig.add_subplot(1, n_plots, j+1)
        ax.set_title("Kmeans Cluster with k="+str(j+2))
        y_lower, y_upper = 0, 0
        yticks = []
        #colores
        colors = ["#5DADE2","#76D7C4","#F7DC6F","#BB8FCE","#F1948A","#BFC9CA"]
        colors = colors[:n_clusters]
        #silhouetee
        for i, cluster in enumerate(cluster_labels):
            cluster_silhouette_vals = silhouette_vals[i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,color=colors[i])
            ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i+1))
            yticks.append((y_lower + y_upper) / 2.)
            y_lower += len(cluster_silhouette_vals)
            
        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Coeficiente de silueta")
        ax.set_ylabel("Cluster")
        #ax.set_xticks(())
        ax.set_yticks(())
    plt.savefig(".\plots\kmeans_clusters.jpeg")
    img=cv2.imread(".\plots\kmeans_clusters.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png


def silhouette_plot_kmedoids(x_list,x_labels,n_plots):
    fig = plt.figure(figsize=(13, 5))
    for j in range(n_plots):
        x=x_list[j]
        labels=x_labels[j]
        silhouette_vals = silhouette_samples(x, labels)
        silhouette_avg = np.mean(silhouette_vals)
        cluster_labels = np.unique(labels)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = [silhouette_vals[labels == i] for i in cluster_labels]
        ax = fig.add_subplot(1, n_plots, j+1)
        ax.set_title("Kmedoids Cluster with k="+str(j+2))
        y_lower, y_upper = 0, 0
        yticks = []
        #colores
        colors = ["#5DADE2","#76D7C4","#F7DC6F","#BB8FCE","#F1948A","#BFC9CA"]
        colors = colors[:n_clusters]
        #silhouetee
        for i, cluster in enumerate(cluster_labels):
            cluster_silhouette_vals = silhouette_vals[i]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,color=colors[i])
            ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i+1))
            yticks.append((y_lower + y_upper) / 2.)
            y_lower += len(cluster_silhouette_vals)
            
        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_xlabel("Coeficiente de silueta")
        ax.set_ylabel("Cluster")
        #ax.set_xticks(())
        ax.set_yticks(())

    plt.savefig(".\plots\kmedoids_clusters.jpeg")
    img=cv2.imread(".\plots\kmedoids_clusters.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

def best_score_silhouette(x_list,x_labels,n_plots):
    best_sil = []
    for j in range(n_plots):
        x=x_list[j]
        labels=x_labels[j]
        silhouette_vals = silhouette_samples(x, labels)
        silhouette_avg = np.mean(silhouette_vals)
        best_sil.append(silhouette_avg)
    print(best_sil)
    return max(best_sil), best_sil.index(max(best_sil))+2
