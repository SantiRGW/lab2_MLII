#FastAPI
from typing import Union
from fastapi import FastAPI,UploadFile
from starlette.responses import StreamingResponse
#Num
import numpy as np
#plots
import cv2
import matplotlib.pyplot as plt
#path
import io
#unsupervised
import modules.kmeans_un as kmeans_un
import modules.kmedoids_un as kmedoids_un
from modules.toy_implement import toy_data,plot_some
#silhouette
from modules.silhouette import silhouette_score,silhouette_coefficient
from sklearn.metrics import silhouette_score, silhouette_samples
#Time
import time
import random

app = FastAPI()


#kmeans_kmedoids
@app.post("/5_original_data_kmeans_kmedoids")
def cluster_distances():
    #Test
    k=4
    x,y,centers = toy_data()
    np.random.seed(1)
    centroids,labels_kmeans  = kmeans_un.kmeans(K=k).fit(x)
    medoids,labels_kmedoids  = kmedoids_un.KMedoids(K=k).fit(x)
    distance_centers=[]
    distance_kmeans=[]
    distance_kmedoids=[]
    for i in range(len(centroids)-1):
        distance_centers.append(np.linalg.norm(centers[0] - centers[i+1]))
        distance_kmeans.append(np.linalg.norm(centroids[0] - centroids[i+1]))
        distance_kmedoids.append(np.linalg.norm(medoids[0] - medoids[i+1]))
    im_png=plot_some(k,labels_kmeans,centroids,x,labels_kmedoids,medoids,y,distance_centers,distance_kmeans,distance_kmedoids)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#kmeans_kmedoids
@app.post("/5_kmeans_kmedoids_silhouette")
def cluster_distances():
    k=4
    x,y,centers = toy_data()
    np.random.seed(1)
    k_means = kmeans_un.kmeans(K=k)
    centroids,cluster_labels  = k_means.fit(x)
    labels = k_means.predict(x)
    silhouette_vals = silhouette_samples(x, labels)
    silhouette_avg = np.mean(silhouette_vals)
    cluster_labels = np.unique(labels)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = [silhouette_vals[labels == i] for i in cluster_labels]
    fig, ax = plt.subplots()
    y_lower, y_upper = 0, 0
    yticks = []

    colors = []
    random.seed(17)
    for _ in range(n_clusters):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
        colors.append(color[0])
    
    for i, cluster in enumerate(cluster_labels):
        cluster_silhouette_vals = silhouette_vals[i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0,color=colors[i])
        ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i+1))
        yticks.append((y_lower + y_upper) / 2.)
        y_lower += len(cluster_silhouette_vals)
        
    ax.axvline(silhouette_avg, color="red", linestyle="--")
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(cluster_labels + 1)
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_ylabel("Cluster")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.show()
    #return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

