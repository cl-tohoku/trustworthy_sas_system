import sys
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.cluster.hierarchy import *
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import collections
import matplotlib.pyplot as plt
from pathlib import Path
import hdbscan


sys.path.append("..")
from library.selector import Selector



class DBSCANAnalyzer:
    def __init__(self):
        self.hierarchy = None

    def clustering(self, attributions, k=10):
        attr = Selector.top_k(attributions)
        attr = MinMaxScaler().fit_transform(attr)
        nearest_neighbors = NearestNeighbors(n_neighbors=20)
        nearest_neighbors.fit(attr)
        distances, indices = nearest_neighbors.kneighbors(attr)
        distances = np.sort(distances, axis=0)[:, 1]
        plt.plot(distances)
        plt.savefig("distance.png")
        db = DBSCAN(eps=5.0, min_samples=5).fit(attr)
        labels = db.labels_
        n_clusters, n_noise = len(set(labels)), list(labels).count(-1)
        print("cluter size:{}, original size: {}, noise size:{}".format(n_clusters, len(labels), n_noise))
        return labels

    def hdbscan(self, attribution):
        attribution = MinMaxScaler().fit_transform(attribution)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        labels = clusterer.fit_predict(attribution)
        n_clusters, n_noise = len(set(labels)), list(labels).count(-1)
        print("cluter size:{}, original size: {}, noise size:{}".format(n_clusters, len(labels), n_noise))

        return labels
