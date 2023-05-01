from model.sequential_kmeans import KMeansSequential
from model.parallel_kmeans import KMeansParallel

def kmeans_factory(k, mode):
    """Return the KMeans algorithm based on the given mode.
    
    Args:
        mode (str): Mode of the algorithm (sequential or parallel).
        k (int): Number of clusters.
        
    Returns:
        KMeansSequential or KMeansParallel: KMeans algorithm.
    """
    return KMeansSequential(k) if mode == 'sequential' else KMeansParallel(k)