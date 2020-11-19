import numpy as np


class KMeansClustering:
    def __init__(self, x, num_clusters, num_iterations=100):
        self.k = num_clusters
        self.iter_num = num_clusters
        self.m, self.n = x.shape
        self.centroids = np.zeros((self.k, self.n))

    def random_initialize_centroids(self, x):  # Initialize central points randomly
        for i in range(self.k):
            self.centroids[i] = x[np.random.choice(range(self.m))]

    def create_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for idx, point in enumerate(X):  # Assign data points to the cluster
            # point (1,n) centroids(k, n)
            closest_centroids = np.argmin(np.sqrt(np.sum((point - self.centroids)**2, axis=1)))
            clusters[closest_centroids].append(idx)

        return clusters

    def cluster_update(self, clusters, x):  # Update central points
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(x[cluster], axis=0)
            self.centroids[idx] = new_centroid

    def predict_label_assign(self, clusters):  # Generate class labels
        y_label = np.zeros(self.m)

        for cluster_idx, cluster in enumerate(clusters):
            for point_idx in cluster:
                y_label[point_idx] = cluster_idx

        return y_label

    def train(self, x):  # Training
        self.random_initialize_centroids(x)

        for i in range(self.iter_num):
            clusters = self.create_clusters(x)
            self.cluster_update(clusters, x)

        y_pred = self.predict_label_assign(clusters, x)

        return y_pred
