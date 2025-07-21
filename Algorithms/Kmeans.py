import random
import numpy as np


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        random_index = random.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for i in range(self.max_iter):
            cluster_group = self.assign_clusters(X)
            # Assign Clusters
            # Move Centroids
            # Check Finish

    def assign_clusters(self, X):
        cluster_group = []
        distances = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row - centroid, row - centroid)))

            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            distances.clear()

        return cluster_group


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

centroids = [(-5, -5), (5, 5)]
cluster_std = [1, 1]

X, y = make_blobs(
    n_samples=100,
    cluster_std=cluster_std,
    centers=centroids,
    n_features=2,
    random_state=2,
)

km = KMeans(n_clusters=2, max_iter=100)
km.fit_predict(X)

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
