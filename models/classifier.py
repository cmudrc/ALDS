import numpy as np
import os
import torch
from sklearn.cluster import KMeans
from joblib import dump, load


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
        self.model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')

    def train(self, data, save_model=False, path=None):
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'kmeans_classifier.joblib'))

    def cluster(self, data):
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'kmeans_classifier.joblib'))