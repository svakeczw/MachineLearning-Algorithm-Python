import numpy as np


class KNearestNeighbor:
    def __init__(self, k, X, y):
        self.k = k
        self.epa = 1e-08  # To prevent from small number error
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, method=1):
        # Select method and make predictions
        if method == 1:
            distances = self.compute_distance_1(X_test)
        if method == 2:
            distances = self.compute_distance_2(X_test)
        if method == 3:
            distances = self.compute_distance_3(X_test)

        return self.predict_labels(distances)

    def compute_distance_1(self, x_test):
        # Method 1: Using two loops to compute Euclidean distance
        num_test = x_test.shape[0]
        num_train = self.X_train.shape[0]
        distance = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                distance[i, j] = np.sqrt(np.sum((self.X_train[j, :] - x_test[i, :])**2))

        return distance

    def compute_distance_2(self, x_test):
        # Method 2: Using one loop to compute Euclidean distance
        num_test = x_test.shape[0]
        num_train = self.X_train.shape[0]
        distance = np.zeros((num_test, num_train))
        for i in range(num_test):
            distance[i, :] = np.sqrt(np.sum((self.X_train - x_test[i, :])**2, axis=1))
        return distance

    def compute_distance_3(self, x_test):
        # Method 3: Using Vectorization to compute Euclidean distance
        distance = np.sqrt(np.sum(self.X_train**2, axis=1, keepdims=True).T + np.sum(x_test**2, axis=1, keepdims=True)
                           - 2*np.dot(x_test, self.X_train.T) + self.epa)
        return distance

    def predict_labels(self, distances):
        # Generate predict labels
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])  # Sort the distance from the nearest to the farthest

            # Find the classes of the K nearest distance samples
            k_closest_classes = self.y_train[y_indices[:self.k], 0].astype(int)

            y_pred[i] = np.argmax(np.bincount(k_closest_classes))  # Take the most frequent class
        return y_pred




