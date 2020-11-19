import numpy as np


class LinearRegressionNormalEquation:
    def __init__(self, X, y):
        """
        :param X: Capital X (m,n) is the design matrix(stack features in a row)
        :param y: Column vector (m,1)
        """
        x0 = np.ones((X.shape[0], 1))
        self.x = np.append(x0, X, axis=1)
        self.y = y
        self.w = None  # Column vector

    def compute(self):
        # cost = (1/2) * np.dot((np.dot(self.x, self.w)-self.y).T, (np.dot(self.x, self.w)-self.y))
        self.w = np.dot(np.linalg.pinv(np.dot(self.x.T, self.x)), np.dot(self.x.T, self.y))
        return self.w


if __name__ == '__main__':
    # Test Normal Equation
    x = np.array([[0,1,2]]).T
    y = np.array([-1,0,1])
    ln = LinearRegressionNormalEquation(x,y)
    w = ln.compute()
    print(w)
    X = np.random.randn(10,2)
    Y = 5 * X[:, 0] + 3 * X[:, 1] + np.random.randn(10,1)*0.1
    LN = LinearRegressionNormalEquation(X, Y)
    W = LN.compute()
    print(W[:])
    print(W.shape)

