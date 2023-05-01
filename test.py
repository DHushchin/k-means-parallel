import numpy as np
from sklearn.datasets import make_blobs
import time

from model.kmeans_factory import kmeans_factory
import matplotlib.pyplot as plt


def generate_data(n_samples, n_features, n_clusters):
    """Generate random data for testing the KMeans algorithm.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        n_clusters (int): Number of clusters to generate.

    Returns:
        tuple: Tuple containing the generated data and the labels.
    """
    return make_blobs(
        n_samples=n_samples, 
        n_features=n_features, 
        centers=n_clusters,
        cluster_std=5.0, 
        shuffle=True, 
        random_state=42
    )[0]


def train_kmeans(train_data, k, mode):
    """Train the KMeans algorithm on the given data.
    
    Args:
        train_data (np.ndarray): Data to train the algorithm on.
        mode (str): Mode of the algorithm (sequential or parallel).
        
    Returns:
        float: Execution time of the algorithm.
    """
    kmeans = kmeans_factory(k, mode)
        
    start_time = time.time()
    kmeans.fit(train_data)
    return time.time() - start_time


def plot_speedup(n_vals, seq_times, par_times):
    """Plot the speedup of the parallel KMeans algorithm compared to the sequential one.

    Args:
        n_vals (list): List of number of samples used for each test.
        seq_times (list): List of execution times for the sequential algorithm.
        par_times (list): List of execution times for the parallel algorithm.
    """
    speedup = [seq_time / par_time for seq_time, par_time in zip(seq_times, par_times)]
    fig, ax = plt.subplots()
    ax.plot(n_vals, speedup)
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Speedup')
    ax.set_title('Parallel KMeans speedup compared to Sequential KMeans')
    
    fig.savefig('results/speedup_vs_samples.png')
    
    
def plot_time(n_vals, seq_times, par_times, val_name):
    """Plot the execution time of the parallel and sequential KMeans algorithms.

    Args:
        n_vals (list): List of test values (sample / clusters / features).
        seq_times (list): List of execution times for the sequential algorithm.
        par_times (list): List of execution times for the parallel algorithm.
        val_name (str): Name of the tested value (sample / clusters / features).
    """
    fig, ax = plt.subplots()
    
    ax.plot(n_vals, seq_times, label='Sequential')
    ax.plot(n_vals, par_times, label='Parallel')
    
    ax.set_xlabel(f'Number of {val_name}')
    ax.set_ylabel('Execution time (s)')
    ax.set_title('Sequential vs Parallel KMeans time comparison')
    
    ax.legend()
    fig.savefig(f'results/time_vs_{val_name}.png')
    

def test_samples():
    """Test the parallel KMeans algorithm with different number of samples.
    """
    print('Testing samples')
    n_clusters = 5
    train_data = generate_data(n_samples=1000000, n_features=2, n_clusters=n_clusters)

    seq_times = []
    par_times = []
    n_vals = tuple(range(100000, 1000001, 100000))
    
    for n in n_vals:
        seq_times.append(train_kmeans(train_data[:n].copy(), n_clusters, 'sequential'))
        par_times.append(train_kmeans(train_data[:n].copy(), n_clusters, 'parallel'))

    plot_time(n_vals, seq_times, par_times, 'samples')
    plot_speedup(n_vals, seq_times, par_times)
    
    
def test_features():
    """Test the parallel KMeans algorithm with different number of features.
    """
    print('Testing features')
    seq_times = []
    par_times = []
    n_features_vals = tuple(range(2, 101, 10))
    
    for n_features in n_features_vals:
        n_clusters = 5
        train_data = generate_data(n_samples=100000, n_features=n_features, n_clusters=n_clusters)
        seq_times.append(train_kmeans(train_data, n_clusters, 'sequential'))
        par_times.append(train_kmeans(train_data, n_clusters, 'parallel'))

    plot_time(n_features_vals, seq_times, par_times, 'features')
    

def test_clusters():
    """Test the parallel KMeans algorithm with different number of clusters.
    """
    print('Testing clusters')
    seq_times = []
    par_times = []
    centers_vals = tuple(range(2, 102, 10))
    
    for centers in centers_vals:
        train_data = generate_data(n_samples=centers * 1000, n_features=2, n_clusters=centers)
        seq_times.append(train_kmeans(train_data, centers, 'sequential'))
        par_times.append(train_kmeans(train_data, centers, 'parallel'))
        
    plot_time(centers_vals, seq_times, par_times, 'clusters')


def test_kmeans():
    """
    Test the sequential and parallel KMeans algorithms on a generated dataset.
    """
    test_samples()
    test_features()
    test_clusters()
