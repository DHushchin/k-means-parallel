import numpy as np


class BaseKMeans:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        
    def _initialize_centroids(self, X):
        indices = np.random.randint(0, X.shape[0], self.k)
        return X[indices]
    
    def fit(self, X):
        pass
    
    def _assign_points_to_centroids(self, X, centroids):
        pass
    
    def _compute_centroids(self, X, labels):
        pass
        
    
    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError('Model has not been fitted yet. Call fit() first.')
        labels = self._assign_points_to_centroids(X, self.centroids)
        return labels
    
    def save_model(self, path):
        np.save(path, self.centroids)
        
    def load_model(self, path):
        self.centroids = np.load(path)
        
    def plot_clusters(self, X):
        pass        
