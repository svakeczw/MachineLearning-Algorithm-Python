import numpy as np
from matplotlib import pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.1, num_iteration=100, lambd=0, decay_rate=0):
        self.alpha = learning_rate
        self.iter_num = num_iteration
        self.m = None
        self.n = None
        self.w = None  # Weight (Theta) (n,1)
        self.b = 0  # Bias
        self.cost_list = []
        self.lambd = lambd  # Regularization lambda
        self.decay = decay_rate  # Learning rate decay rate

    def yhat(self, x):
        # x(n, m) w(n,1)
        yhat = np.dot(self.w.T, x)
        assert yhat.shape == (1, x.shape[1])
        return yhat

    def train(self, x, y):
        # y(1, m)
        x0 = np.ones((1, x.shape[1]))
        x = np.append(x, x0, axis=0)
        assert x.shape[1] == y.shape[1]

        self.m = x.shape[1]
        self.n = x.shape[0]
        self.w = np.zeros((self.n, 1))

        # Gradient descent
        for i in range(self.iter_num):
            y_hat = self.yhat(x)

            cost = (1 / (2 * self.m)) * np.sum(np.power((y_hat - y), 2))
            self.cost_list.append(cost)

            dw = (1 / self.m) * np.dot(x, (y_hat - y).T)

            self.w -= self.alpha * dw

            if i % 100 == 0 and i != 0:
                print(f'Cost after {i} iteration is {cost}')
        print(f'Cost after {self.iter_num} iteration is {cost}')
        return self.w

    def plot_cost(self):
        # Plot cost curve
        plt.plot(range(self.iter_num), self.cost_list)
        plt.xlabel('Iteration Number')
        plt.ylabel('Cost')
        plt.show()







