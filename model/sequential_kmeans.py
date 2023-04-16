import numpy as np

from model.base_kmeans import BaseKMeans


class KMeansSequential(BaseKMeans):
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        super().__init__(k, max_iter, tol, random_state)

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
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        centroids = []
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                centroid = np.random.rand(X.shape[1])
            centroids.append(centroid)

        return np.array(centroids)
