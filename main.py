from test import test_kmeans

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time

from model.kmeans_factory import kmeans_factory


def run(n_samples, n_features, n_clusters, visualize=False):
    """Run the KMeans algorithm on randomly generated data.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_clusters (int): Number of clusters to generate.
        visualize (bool, optional): Whether to visualize the results. Defaults to False.
    """
    train_data, _ = make_blobs (
        n_samples=n_samples, 
        n_features=n_features, 
        centers=n_clusters, 
        cluster_std=5.0, 
        shuffle=True, 
        random_state=42
    )

    kmeans_seq = kmeans_factory('sequential', k=n_clusters)
    start_time = time.time()
    kmeans_seq.fit(train_data)
    print(f"Sequential KMeans time: {time.time() - start_time:.2f} seconds")
    kmeans_seq.save_model('model/weights/sequential_kmeans.npy')
    pred1 = kmeans_seq.predict(train_data)

    kmeans_par = kmeans_factory('parallel', k=n_clusters)
    start_time = time.time()
    kmeans_par.fit(train_data)
    print(f"Parallel KMeans time: {time.time() - start_time:.2f} seconds")
    kmeans_par.save_model('model/weights/parallel_kmeans.npy')
    pred2 = kmeans_par.predict(train_data)

    print(f"Are the predictions equal? {np.array_equal(pred1, pred2)}")
    
    if visualize:
        kmeans_seq.plot_clusters(train_data, 'Sequential KMeans')
        kmeans_par.plot_clusters(train_data, 'Parallel KMeans')
        plt.show()

if __name__ == '__main__':
    # run(100000, 3, 3, visualize=True)
    test_kmeans()
