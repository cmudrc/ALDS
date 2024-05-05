import numpy as np
import os
import torch
from sklearn.cluster import KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
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
        self.model = KMeansWasserstein(n_clusters=n_clusters, random_state=0)

    def train(self, data, save_model=False, path=None):
        data = self.scaler.fit_transform(data)
        self.model.fit(data)
        if save_model:
            self._save_model(path)

    def _save_model(self, path):
        dump(self.model, os.path.join(path, 'wasserstein_kmeans_classifier.joblib'))
        dump(self.scaler, os.path.join(path, 'wasserstein_kmeans_scaler.joblib'))
             
    def load_model(self, path):
        self.model = load(os.path.join(path, 'wasserstein_kmeans_classifier.joblib'))
        self.scaler = load(os.path.join(path, 'wasserstein_kmeans_scaler.joblib'))
                           
    def cluster(self, data):
        data = self._normalize(data)
        return self.model.predict(data)
    

class KMeansWasserstein(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, distance_metric="euclidean"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.distance_metric = distance_metric

    def fit(self, X, y=None):
        n_samples, n_features = X.shape

        rng = check_random_state(self.random_state)
        self.labels_ = rng.randint(self.n_clusters, size=n_samples)

        best_inertia = None
        for _ in range(self.max_iter):
            centers = self._calculate_centers(X)
            distances = self._compute_distance(X, centers)
            labels = np.argmin(distances, axis=1)

            if np.sum(labels != self.labels_) == 0:
                break

            self.labels_ = labels

            inertia = np.sum(np.min(distances, axis=1))
            if best_inertia is None or inertia < best_inertia - self.tol:
                best_inertia = inertia
            else:
                break

        self.cluster_centers_ = self._calculate_centers(X)
        self.inertia_ = best_inertia

        return self

    def _calculate_centers(self, X):
        centers = np.empty((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            mask = self.labels_ == i
            if np.sum(mask) == 0:
                # Empty cluster, choose a random point
                centers[i] = X[np.random.randint(X.shape[0])]
            else:
                # Compute the average of Wasserstein distances to all points in the cluster
                cluster_points = X[mask]
                w_distances = pairwise_distances(cluster_points, metric=self._wasserstein_distance)
                mean_point = np.mean(w_distances, axis=0)
                centers[i] = cluster_points[np.argmin(mean_point)]

        return centers

    def _compute_distance(self, X, centers):
        if self.distance_metric == "euclidean":
            return pairwise_distances(X, centers, metric="euclidean")
        elif self.distance_metric == "wasserstein":
            return pairwise_distances(X, centers, metric=self._wasserstein_distance)

    def _wasserstein_distance(self, x, y):
        return wasserstein_distance(x, y)

    def predict(self, X):
        distances = self._compute_distance(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)