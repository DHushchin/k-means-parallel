import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple
from threading import Thread, Lock
from model.base import BaseSOM


class ParallelSOM(BaseSOM):
    def __init__(self, map_size: Tuple[int, int], n_features: int, learning_rate: float):
        self.map_size = map_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        np.random.seed(42)
        self.weights = np.random.uniform(0, 1, (map_size[0], map_size[1], n_features))

    def train(self, X, n_epochs, n_threads=1):
        n_samples = X.shape[0]
        lock = Lock() 

        def thread_func(start, end):
            for idx in range(start, end):
                x = X[idx]
                bmu = self.find_bmu(x)
                self.update_weights(x, bmu, epoch, n_epochs, lock)

        for epoch in range(n_epochs):

            if n_threads == 1:
                thread_func(0, n_samples)
            else:
                threads = []
                batch_size = n_samples // n_threads
                
                for i in range(n_threads):
                    start = i * batch_size
                    end = start + batch_size if i < n_threads - 1 else n_samples
                    t = Thread(target=thread_func, args=(start, end))
                    threads.append(t)

                for t in threads:
                    t.start()

                for t in threads:
                    t.join()


    def update_weights(self, x, bmu, epoch, n_epochs, lock):
        sigma = self.calculate_sigma(epoch, n_epochs)
        lr = self.calculate_learning_rate(epoch, n_epochs)
        neighbourhood = self.calculate_neighbourhood(bmu, sigma)

        with lock:
            self.weights += lr * neighbourhood[:, :, np.newaxis] * (x - self.weights)


    def find_bmu(self, x):
        distances = cdist(
            x.reshape(1, self.n_features), 
            self.weights.reshape(self.map_size[0] * self.map_size[1], 
            self.n_features)
        )
        bmu = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu


    def calculate_sigma(self, epoch, n_epochs):
        return self.map_size[0] / 2 * (1 - epoch / n_epochs)
    

    def calculate_learning_rate(self, epoch, n_epochs):
        return self.learning_rate * (1 - epoch / n_epochs)
    

    def calculate_neighbourhood(self, bmu, sigma):
        neighborhood = np.zeros(self.map_size)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                neighborhood[i, j] = np.exp(-dist**2 / (2*sigma**2))
        
        return neighborhood
  

    def predict(self, X):
        return self.find_bmu(X)
    
    def save_weights(self, filename):
        np.save(filename, self.weights)
  
    def load_weights(self, filename):
        self.weights = np.load(filename)
