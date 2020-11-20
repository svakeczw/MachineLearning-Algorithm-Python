import numpy as np
import cvxopt


class SVM:
    def __init__(self, c=1):
        self.kernel = None
        self.C = c
        self.alpha = None
        self.w = None
        self.b = None
        self.X = None
        self.y = None

    @staticmethod
    def gaussian(x, z, sigma=0.1):
        return np.exp(-np.linalg.norm(x-z, axis=1)**2 / (2*(sigma**2)))

    @staticmethod
    def linear(x, z):
        return np.dot(x, z.T)

    @staticmethod
    def polynomial(x, z, degree=3):
        return (1 + np.dot(x, z.T)) ** degree

    def train(self, x, y, kernal='gaussian'):
        self.X = x
        self.y = y
        m, n = self.X.shape

        self.kernel = np.zeros((m, m))
        for i in range(m):
            if kernal == 'gaussian':
                self.kernel[i, :] = self.gaussian(x[i, np.newaxis], self.X)
            elif kernal == 'linear':
                self.kernel[i, :] = self.linear(x[i, np.newaxis], self.X)
            elif kernal == 'polynomial':
                self.kernel[i, :] = self.polynomial(x[i, np.newaxis], self.X)

        # transform to cvxopt format
        P = cvxopt.matrix(np.outer(y, y) * self.kernel)  # H(i,j) = y(i)y(j)<x(i)x(j)>
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m)*self.C)))  # 0 < alpha < C
        A = cvxopt.matrix(y, (1, m), 'd')  # y label vector
        b = cvxopt.matrix(np.zeros(1))  # bias scalar
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x'])

    def predict(self, x):
        y_pred = np.zeros((x.shape[0]))
        sv = self.get_parameters(self.alpha)
        for i in range(x.shape[0]):

            # ∑i,jmy(i)y(j)αiαj<x(i)x(j)>
            y_pred[i] = np.sum(
                self.alpha[sv]
                * self.y[sv, np.newaxis]
                * self.gaussian(x[i], self.X[sv])[:, np.newaxis]
            )

        return np.sign(y_pred + self.b)  # y′=sign(wTx′+b)

    def get_parameters(self, alphas):
        threshold = 1e-5

        # support vectors
        sv = ((alphas > threshold) * (alphas < self.C)).flatten()
        self.w = np.dot(self.X[sv].T, alphas[sv] * self.y[sv, np.newaxis])
        self.b = np.mean(
            self.y[sv, np.newaxis]
            - self.alpha[sv]
            * self.y[sv, np.newaxis]
            * self.kernel[sv, sv][:, np.newaxis]
        )
        return sv


# Reference:
# https://xavierbourretsicotte.github.io/SVM_implementation.html
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/algorithms/svm/svm.py
