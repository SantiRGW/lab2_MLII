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
#kmeans unsupervised
import modules.kmeans_un as kmeans_un
from modules.toy_impliment import toy_data,plot_toy
#Time
import time
app = FastAPI()

#Rectangular matrix
@app.post("/5_Kmeans")
def kmeans_test():
    #Test
    from sklearn.cluster import KMeans
    x,y = toy_data()
    print(x)
    plot_toy(x,y)
    km = kmeans_un.kmeans(K=4).fit(x)
    centros=km
    print(centros)
    plt.scatter(centros[:,0],centros[:,0],c='r')
    plt.savefig(".\plots\plot_globs.jpeg")
    img=cv2.imread(".\plots\plot_globs.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpeg")