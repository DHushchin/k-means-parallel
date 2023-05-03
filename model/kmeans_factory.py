from model.sequential_kmeans import KMeansSequential
from model.parallel_kmeans import KMeansParallel

def kmeans_factory(k, mode, n_threads=8):
    """Return the KMeans algorithm based on the given mode.
    
    Args:
        mode (str): Mode of the algorithm (sequential or parallel).
        k (int): Number of clusters.
        n_threads (int, optional): Number of threads to use for the parallel algorithm. Defaults to 8.
        
    Returns:
        KMeansSequential or KMeansParallel: KMeans algorithm.
    """
    return KMeansSequential(k, n_threads) if mode == 'sequential' else KMeansParallel(k, n_threads)