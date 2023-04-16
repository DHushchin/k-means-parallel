import numpy as np
import matplotlib.pyplot as plt
import threading

from model.base_kmeans import BaseKMeans

class KMeansParallel(BaseKMeans):
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42, n_threads=4):
        super().__init__(k, max_iter, tol, random_state)
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

        threads = []
        for i in range(self.n_threads):
            start_idx = i * step
            end_idx = start_idx + step if i < self.n_threads - 1 else X.shape[0]
            t = threading.Thread(target=assign_points_thread, args=(start_idx, end_idx))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

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

        threads = []
        for i in range(self.k):
            t = threading.Thread(target=compute_centroids_thread, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return centroids
    
        
    def plot_clusters(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.predict(X), s=40, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', s=200)
        plt.title('Parallel K-Means')
        plt.show()
