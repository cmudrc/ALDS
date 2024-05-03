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


class WassersteinKMeansClassifier(KMeansClassifier):
    def __init__(self, n_clusters):
        super(WassersteinKMeansClassifier, self).__init__(n_clusters)
        
    def _wasserstein_distance(self, x1, x2, p=2):
        F1 = np.cumsum(x1)
        F2 = np.cumsum(x2)

        return np.sum(np.abs(F1 - F2) ** p) ** (1 / p)
    
    def _kplusplus(self, data, n_clusters):
        n_samples, n_features = data.shape
        centers = np.zeros((n_clusters, n_features))
        centers[0] = data[np.random.choice(n_samples)]
        distances = np.zeros(n_samples)
        for i in range(1, n_clusters):
            for j in range(n_samples):
                distances[j] = np.min([self._wasserstein_distance(data[j], centers[k]) for k in range(i)])
            centers[i] = data[np.argmax(distances)]
        return centers
    
    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.cluster_centers_ = self._kplusplus(data, self.n_clusters)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'wasserstein_kmeans_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'wasserstein_kmeans_scaler.joblib'))