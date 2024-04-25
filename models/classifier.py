import numpy as np
import os
import torch
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


class Classifier:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def train(self, data):
        pass

    def _normalize(self, data):
        return self.scaler.transform(data)

    def cluster(self, data):
        pass


class KMeansClassifier(Classifier):
    def __init__(self, n_clusters):
        super(KMeansClassifier, self).__init__(n_clusters)
        self.model = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'kmeans_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'kmeans_scaler.joblib'))

    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'kmeans_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'kmeans_scaler.joblib'))


class MeanShiftClassifier(Classifier):
    def __init__(self):
        super(MeanShiftClassifier, self).__init__(n_clusters=None)
        self.model = MeanShift(cluster_all=True, n_jobs=-1)

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        self.n_clusters = len(np.unique(self.model.labels_))
        print(f'Mean shift identified {self.n_clusters} clusters')
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'mean_shift_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'mean_shift_scaler.joblib'))

    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'mean_shift_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'mean_shift_scaler.joblib'))


class GaussianMixtureClassifier(Classifier):
    def __init__(self, n_clusters):
        super(GaussianMixtureClassifier, self).__init__(n_clusters)
        self.model = GaussianMixture(n_components=n_clusters, random_state=0)

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'gmm_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'gmm_scaler.joblib'))

    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    
    def load_model(self, path):
        self.model = load(os.path.join(path, 'gmm_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'gmm_scaler.joblib'))