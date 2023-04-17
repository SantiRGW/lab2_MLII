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
#Time
import time

app = FastAPI()


#kmeans_kmedoids
@app.post("/5_original_data_kmeans_kmedoids")
def cluster_distances():
    #Test
    k=4
    x,y,centers = toy_data()
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


