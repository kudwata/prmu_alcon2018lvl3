import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def clusterring(images, N_CLUSTERS=64):

    cluster = KMeans(n_clusters=N_CLUSTERS, n_jobs=-1)
    predict = cluster.fit_predict(images)

    return predict