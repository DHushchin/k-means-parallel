import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from model.base_kmeans import BaseKMeans


class KMeansParallel(BaseKMeans):
    def __init__(self, k, max_iter=100, random_state=42, n_threads=8):
        super().__init__(k, max_iter, random_state)
        self.n_threads = n_threads

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        centroids = self._initialize_centroids(X)
        labels = None

        for _ in range(self.max_iter):
            labels = self._assign_points_to_centroids(X, centroids)    
            centroids = self._compute_centroids(X, labels)
            
            self.centroids = centroids

    def _assign_points_to_centroids(self, X, centroids):
        labels = np.zeros(X.shape[0], dtype=int)
        step = X.shape[0] // self.n_threads

        def assign_points_thread(start_idx, end_idx):
            distances = np.linalg.norm(X[start_idx:end_idx, np.newaxis, :] - centroids, axis=2)
            labels[start_idx:end_idx] = np.argmin(distances, axis=1)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(self.n_threads):
                start_idx = i * step
                end_idx = start_idx + step
                executor.submit(assign_points_thread, start_idx, end_idx)

        return labels
    
    def _compute_centroids(self, X, labels):
        centroids = np.zeros((self.k, X.shape[1]))

        def compute_centroids_thread(i):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                centroid = np.random.rand(X.shape[1])
            centroids[i] = centroid

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            for i in range(self.k):
                executor.map(compute_centroids_thread, range(self.k))
        
        return centroids
