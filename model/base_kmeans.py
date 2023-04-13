class BaseKMeans:
    def _compute_centroids(self, X, labels):
        pass

    def _compute_distances(self, X, centroids):
        pass

    def _assign_labels(self, X, centroids):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
