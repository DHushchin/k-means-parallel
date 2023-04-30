import numpy as np
import matplotlib.pyplot as plt

class BaseKMeans:
    def __init__(self, k: int = 3, max_iter: int = 100, random_state: int = 42):
        """Base class for the KMeans algorithm.

        Args:
            k (int, optional): Number of clusters. Defaults to 3.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            random_state (int, optional): _description_. Defaults to 42.
        """
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes the centroids.

        Args:
            X (np.ndarray): Data points.

        Returns:
            np.ndarray: Centroids.
        """
        indices = np.random.randint(0, X.shape[0], self.k)
        return X[indices]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the cluster for each data point.
        
        Args:
            X (np.ndarray): Data points.
            
        Returns:
            np.ndarray: Cluster for each data point.
        """
        if self.centroids is None:
            raise RuntimeError('Model has not been fitted yet. Call fit() first.')
        labels = self._assign_points_to_centroids(X, self.centroids)
        return labels
    
    def plot_clusters(self, X: np.ndarray, title: str):
        """Plots the clusters.

        Args:
            X (np.ndarray): Data points.
            title (str): Title of the plot.
        """
        plt.scatter(X[:, 0], X[:, 1], c=self.predict(X), s=40, cmap='viridis')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', s=200)
        plt.title(title)
        plt.show()  
    
    def fit(self, X: np.ndarray):
        """Fits the model.

        Args:
            X (np.ndarray): Data points.
        """
        pass
    
    def _assign_points_to_centroids(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assigns each data point to the closest centroid.

        Args:
            X (np.ndarray): Data points.
            centroids (np.ndarray): Centroids.

        Returns:
            np.ndarray: Cluster for each data point.
        """
        pass
    
    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Computes the centroids.
        
        Args:
            X (np.ndarray): Data points.
            
        Returns:
            np.ndarray: Centroids.
        """
        pass
    
    def save_model(self, path: str):
        """Saves the model to a file.

        Args:
            path (str): Path to the file.
        """
        np.save(path, self.centroids)
        
    def load_model(self, path: str):
        """Loads the model from a file.

        Args:
            path (str): Path to the file.
        """
        self.centroids = np.load(path)
