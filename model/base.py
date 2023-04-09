from typing import Tuple

class BaseSOM:
    def __init__(self, map_size: Tuple[int, int], n_features: int, learning_rate: float):
        pass

    def train(self, X, n_epochs):
        pass

    def find_bmu(self, x):
        pass

    def update_weights(self, x, bmu, epoch, n_epochs):
        pass

    def calculate_sigma(self, epoch, n_epochs):
        pass

    def calculate_learning_rate(self, epoch, n_epochs):
        pass

    def calculate_neighbourhood(self, bmu, sigma):
        pass

    def predict(self, X):
        pass

    def save_weights(self, filename):
        pass

    def load_weights(self, filename):
        pass
