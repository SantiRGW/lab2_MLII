from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import cv2

def toy_data():
    return make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
    )
def plot_toy(x,y):
    plt.scatter(x[:, 0], x[:, 1], marker="o", c=y)