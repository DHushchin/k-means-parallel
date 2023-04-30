from model.sequential_kmeans import KMeansSequential
from model.parallel_kmeans import KMeansParallel

def kmeans_factory(mode):
    """Return the KMeans algorithm based on the given mode.
    
    Args:
        mode (str): Mode of the algorithm (sequential or parallel).
        
    Returns:
        KMeansSequential or KMeansParallel: KMeans algorithm.
    """
    return KMeansSequential() if mode == 'sequential' else KMeansParallel()