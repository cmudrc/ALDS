import numpy as np
import os
import torch
from sklearn.cluster import KMeans


class Classifier:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def train(self, data):
        pass

    def cluster(self, data):
        pass


class KMeansClassifier(Classifier):
    def __init__(self, n_clusters):
        super(KMeansClassifier, self).__init__(n_clusters)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')

    def train(self, data):
        self.kmeans.fit(data)

    def cluster(self, data):
        return self.kmeans.predict(data)