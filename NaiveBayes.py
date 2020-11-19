import numpy as np


class NaiveBayes:
    def __init__(self, x, y):
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.classes = np.unique(y)
        self.num_class = len(self.classes)
        self.mean = np.zeros((self.num_class, self.n), dtype=np.float64)
        self.var = np.zeros((self.num_class, self.n), dtype=np.float64)
        self.priors = np.zeros(self.num_class, dtype=np.float64)

    def train(self, x, y):
        # Calculate mean, variance, prior for all classes
        for idx,c in enumerate(self.classes):
            X_c = x[c == y]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0)
            self.priors[idx] = X_c.shape[0] / float(self.m)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probability for all classes
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._prior_density_function(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]  # Get the highest probability

    def _prior_density_function(self, c_idx, x):
        mean = self.mean[c_idx]
        var = self.var[c_idx]
        numerator = np.exp(-(x - mean)**2 / 2 * var)
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


