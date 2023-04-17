from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import cv2
import random
#Creating toy data
def toy_data():
    return make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
    return_centers=True
    )

#plot kmeans clustering
def plot_some(n_clusters,k_means_labels,k_means_cluster_centers,X,k_medoids_labels,k_medoids_cluster_centers,y,distance_centers,distance_kmeans,distance_kmedoids):
    fig = plt.figure(figsize=(9, 5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = []
    random.seed(17)
    for _ in range(n_clusters):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
        colors.append(color[0])
    #original
    ax = fig.add_subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="v", c=y)
    ax.set_title("Original data")
    ax.set_xticks(())
    ax.set_yticks(())
    for i in range(len(distance_centers)):
        plt.text(-13, 4.8 + i, "Dist 1 to "+str(-i+4)+" is: "+ str(round(distance_centers[i],2)))

    # KMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="v")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMeans")
    ax.set_xticks(())
    ax.set_yticks(())
    for i in range(len(distance_kmeans)):
        plt.text(-13, 4.8 + i, "Dist 1 to "+str(-i+4)+" is: "+ str(round(distance_kmeans[i],2)))

    # KMedoids
    ax = fig.add_subplot(1, 3, 3)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_medoids_labels == k
        cluster_center = k_medoids_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker="v")
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("KMedoids")
    ax.set_xticks(())
    ax.set_yticks(())
    for i in range(len(distance_kmedoids)):
        plt.text(-13, 4.8 + i, "Dist 1 to "+str(-i+4)+" is: "+ str(round(distance_kmedoids[i],2)))

    plt.savefig(".\plots\plot_globs.jpeg")
    img=cv2.imread(".\plots\plot_globs.jpeg")
    res, im_png = cv2.imencode(".jpeg", img)
    return im_png