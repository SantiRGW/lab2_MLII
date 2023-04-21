from modules.toy_implement import toy_data_2
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
import cv2

data = toy_data_2()

#Plot and apply of kmeans sklearn
def kmeans_ls_result(k=4):
    lista_labels_dat=["noisy_circles","noisy_moons","blobs","no_structure","aniso","varied"]
    fig = plt.figure(figsize=(17, 5))
    k=k     #clusters
    i=0
    for d in data:
        X=d[0]
        y=d[1]
        #kmeans
        ax = fig.add_subplot(1, len(data), i+1)
        ax.set_title("Plot "+lista_labels_dat[i])
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        y_pred = kmeans.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
        #plt.title('KMeans Clustering')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.legend()
        i+=1
    plt.savefig(".\plots\plot_dif_data_kmeans_ls.jpeg")
    img=cv2.imread(".\plots\plot_dif_data_kmeans_ls.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

#Plot and apply of kmedoids sklearn
def kmedoids_ls_result(k=4):
    lista_labels_dat=["noisy_circles","noisy_moons","blobs","no_structure","aniso","varied"]
    fig = plt.figure(figsize=(17, 5))
    k=k     #clusters
    i=0
    for d in data:
        X=d[0]
        y=d[1]
        #kmeans
        ax = fig.add_subplot(1, len(data), i+1)
        ax.set_title("Plot "+lista_labels_dat[i])
        kmedoids = KMedoids(n_clusters=k, random_state=0).fit(X)
        y_pred = kmedoids.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=100, c='red', label='Medoid')
        #plt.title('KMeans Clustering')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.legend()
        i+=1
    plt.savefig(".\plots\plot_dif_data_kmedoid_ls.jpeg")
    img=cv2.imread(".\plots\plot_dif_data_kmedoid_ls.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

#Plot and apply of DBSCAN sklearn
def DBSCAN_ls_result():
    lista_labels_dat=["noisy_circles","noisy_moons","blobs","no_structure","aniso","varied"]
    fig = plt.figure(figsize=(17, 5))
    i=0
    for d in data:
        X=d[0]
        y=d[1]
        #kmeans
        ax = fig.add_subplot(1, len(data), i+1)
        ax.set_title("Plot "+lista_labels_dat[i])
        clustering = DBSCAN(eps=0.2, min_samples=10).fit(X)
        y_pred = clustering.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        labels = clustering.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        plt.title('DBSCAN: '+str(n_clusters_)+' clusters')
        ax.set_xticks(())
        ax.set_yticks(())
        i+=1
    plt.savefig(".\plots\plot_dif_data_DBSCAN_ls.jpeg")
    img=cv2.imread(".\plots\plot_dif_data_DBSCAN_ls.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png

#Plot and apply of SpectralClustering sklearn
def SpectralClustering_ls_result(k=4):
    lista_labels_dat=["noisy_circles","noisy_moons","blobs","no_structure","aniso","varied"]
    fig = plt.figure(figsize=(17, 5))
    k=k     #clusters
    i=0
    for d in data:
        X=d[0]
        y=d[1]
        #kmeans
        ax = fig.add_subplot(1, len(data), i+1)
        ax.set_title("Plot "+lista_labels_dat[i])
        spectralClustering = SpectralClustering(n_clusters=k,eigen_solver="arpack",affinity="nearest_neighbors",).fit(X)
        y_pred = spectralClustering.fit_predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        ax.set_xticks(())
        ax.set_yticks(())
        i+=1
    plt.savefig(".\plots\plot_dif_data_SpectralClustering_ls.jpeg")
    img=cv2.imread(".\plots\plot_dif_data_SpectralClustering_ls.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png