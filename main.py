import numpy as np
from sklearn.datasets import make_blobs
import time

from model.sequential_kmeans import KMeansSequential
from model.parallel_kmeans import KMeansParallel


def main():
    train_data, _ = make_blobs(n_samples=1000000, n_features=10, centers=3, cluster_std=1.0, shuffle=True, random_state=42)

    start_time = time.time()
    kmeans_seq = KMeansSequential(k=3)
    kmeans_seq.fit(train_data)
    print(f"Sequential KMeans time: {time.time() - start_time:.2f} seconds")
    kmeans_seq.save_model('model/weights/sequential_kmeans.npy')
    pred1 = kmeans_seq.predict(train_data)
    

    start_time = time.time()
    kmeans_par = KMeansParallel(k=3)
    kmeans_par.fit(train_data)
    print(f"Parallel KMeans time: {time.time() - start_time:.2f} seconds")
    kmeans_par.save_model('model/weights/parallel_kmeans.npy')
    pred2 = kmeans_par.predict(train_data)
    
    print(f"Are the predictions equal? {np.array_equal(pred1, pred2)}")


if __name__ == '__main__':
    main()
