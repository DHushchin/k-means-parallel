import os
import numpy as np
from threading import Thread
from model.base_kmeans import BaseKMeans


class KMeansParallel(BaseKMeans):
    def __init__(self, k, n_iters=100, n_threads=os.cpu_count()):
        self.k = k
        self.n_iters = n_iters
        self.n_threads = n_threads
        self.centroids = None
        print(f"Using {self.n_threads} threads")
    
    def _compute_centroids(self, X, labels):
        centroids = []
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                np.random.seed(42)
                centroid = np.random.rand(X.shape[1])
            centroids.append(centroid)
        return np.array(centroids)
    
    def _compute_distances(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, X, centroids):
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _fit_thread(self, X, start_idx, end_idx, labels):
        for _ in range(self.n_iters):
            centroids = self._compute_centroids(X[start_idx:end_idx], labels[start_idx:end_idx])
            labels[start_idx:end_idx] = self._assign_labels(X[start_idx:end_idx], centroids)
    
    def fit(self, X):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        np.random.seed(42)
        self.centroids = np.random.rand(self.k, X.shape[1])
        
        chunk_size = n_samples // self.n_threads
        threads = []
        
        for i in range(self.n_threads):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.n_threads - 1 else n_samples
            t = Thread(target=self._fit_thread, args=(X, start_idx, end_idx, labels))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.labels_ = labels

    def predict(self, X):
        return self._assign_labels(X, self.centroids)
    
    def save_model(self, filename):
        np.save(filename, self.centroids)

    def load_model(self, filename):
        self.centroids = np.load(filename)
