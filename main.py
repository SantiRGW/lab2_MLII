#FastAPI
from typing import Union
from fastapi import FastAPI,UploadFile
from starlette.responses import StreamingResponse
#Num
import numpy as np
#path
import io
#unsupervised
import modules.kmeans_un as kmeans_un
import modules.kmedoids_un as kmedoids_un
from modules.toy_implement import toy_data,plot_some,plot_toy_data_2
#silhouette
from modules.silhouette import silhouette_plot_kmeans,silhouette_plot_kmedoids,best_score_silhouette
#scikit-learn
from modules.model_sl import kmeans_ls_result,kmedoids_ls_result,DBSCAN_ls_result,SpectralClustering_ls_result

app = FastAPI()

#Original_kmeans_kmedoids
@app.post("/5_original_data_kmeans_kmedoids")
def cluster_distances():
    #Test
    k=4
    x,y,centers = toy_data()
    np.random.seed(2)
    centroids,labels_kmeans  = kmeans_un.kmeans(K=k).fit(x)
    medoids,labels_kmedoids  = kmedoids_un.KMedoids(n_clusters=k).fit(x)
    distance_centers=[]
    distance_kmeans=[]
    distance_kmedoids=[]
    #To calculate distances
    for i in range(len(centroids)-1):
        distance_centers.append(np.linalg.norm(centers[0] - centers[i+1]))
        distance_kmeans.append(np.linalg.norm(centroids[0] - centroids[i+1]))
        distance_kmedoids.append(np.linalg.norm(medoids[0] - medoids[i+1]))
    #Plot of results original data, kmeans and kmedoids
    im_png=plot_some(k,labels_kmeans,centroids,x,labels_kmedoids,medoids,y,distance_centers,distance_kmeans,distance_kmedoids)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#kmeans silhouette
@app.get("/5_kmeans_kmeans_silhouette")
def kmeans_silhouette(K= 5):
    x_list=[]
    x_labels=[]
    n_test=int(K)    #numero de clusters + 1
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_means = kmeans_un.kmeans(K=k)
        centroids,cluster_labels  = k_means.fit(x)
        labels = k_means.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    #plot kmeans solhouette
    im_png = silhouette_plot_kmeans(x_list,x_labels,n_test)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#kmedoids silhouette
@app.get("/5_kmeans_kmedoids_silhouette")
def kmedoids_silhouette(K = 5):
    x_list=[]
    x_labels=[]
    n_test=int(K)   #numero de clusters + 1
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_medoids = kmedoids_un.KMedoids(n_clusters=k)
        centroids,cluster_labels  = k_medoids.fit(x)
        labels = k_medoids.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    #plot kmedoids solhouette
    im_png = silhouette_plot_kmedoids(x_list,x_labels,n_test)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#score silhouette
@app.get("/5_score_silhouette")
def score_silhouette(K = 5):
    #best_score_silhoutte_kmeans
    x_list=[]
    x_labels=[]
    n_test=int(K)    #numero de clusters + 1
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_means = kmeans_un.kmeans(K=k)
        centroids,cluster_labels  = k_means.fit(x)
        labels = k_means.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    best_sil_kmeans,n_clus_means = best_score_silhouette(x_list,x_labels,n_test)

    #best_score_silhoutte_kmedoids
    x_list=[]
    x_labels=[]
    n_test=int(K)    #numero de clusters + 1
    for i in range(n_test):
        k=i+2
        x,y,centers = toy_data()
        np.random.seed(2)
        k_medoids = kmedoids_un.KMedoids(n_clusters=k)
        centroids,cluster_labels  = k_medoids.fit(x)
        labels = k_medoids.predict(x)
        x_list.append(x)
        x_labels.append(labels)
    #best score for silhouette
    best_sil_kmedoids, n_clus_medoids = best_score_silhouette(x_list,x_labels,n_test)
    return {"the best silhouette score in kmeans is " +str(best_sil_kmeans)+ " for clusters" : n_clus_means,
            "the best silhouette score in kmedoids is  "+str(best_sil_kmedoids)+ " for clusters" : n_clus_medoids}

#plot_different_data
@app.post("/6_plot_different_data")
def plot_dif_data():
    im_png=plot_toy_data_2()
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#Scikit-Learn_kmeans
@app.get("/6_Apply_Scikit-Learn_kmeans_to_data")
def plot_dif_data_kmeans(K = 3):
    im_png=kmeans_ls_result(k=int(K))
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#Scikit-Learn_kmedoids
@app.get("/6_Apply_Scikit-Learn_kmedoids_to_data")
def plot_dif_data_kmedoids(K = 3):
    im_png=kmedoids_ls_result(k=int(K))
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#Scikit-Learn_DBSCAN
@app.post("/6_Apply_Scikit-Learn_DBSCAN_to_data")
def plot_dif_data_DBSCAN():
    im_png=DBSCAN_ls_result()
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")

#Scikit-Learn_DBSCAN
@app.get("/6_Apply_Scikit-Learn_SpectralClustering_to_data")
def plot_dif_data_SpectralClustering(K = 2):
    im_png=SpectralClustering_ls_result(k=int(K))
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")