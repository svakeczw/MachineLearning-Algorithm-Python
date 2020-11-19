import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, x, learning_rate=0.01, num_iteration=100, lambd=0, decay_rate=0):
        self.alpha = learning_rate
        self.iter_num = num_iteration
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.w = np.zeros((self.n, 1))  # Weight (Theta)
        self.b = 0  # Bias
        self.cost_list = []
        self.lambd = lambd  # Regularization lambda
        self.decay = decay_rate  # Learning rate decay rate

    def sigmoid(self, z):
        # Sigmoid function
        s = 1 / (1 + np.exp(-z))
        return s

    def train(self, x, y):
        # Gradient decent
        for i in range(self.iter_num):
            y_hat = self.sigmoid(np.dot(x, self.w) + self.b)

            cost = (-1 / self.m) * np.sum(y*np.log(y_hat) + (1-y) * np.log(1-y_hat))
            assert cost.shape == ()
            self.cost_list.append(cost)

            dw = (1 / self.m) * np.dot(x.T, (y_hat - y))
            db = (1 / self.m) * np.sum(y - y_hat)
            assert dw.shape == (x.shape[1], 1)
            assert db.shape == ()

            self.w -= self.alpha * dw + (self.lambd / self.m) * self.w
            self.b -= self.alpha * db

            self.alpha = (1/(1+self.decay * i)) * self.alpha

            if i % 100 == 0 and i != 0:
                print(f'Cost after {i} iteration is {cost}')
        print(f'Cost after {self.iter_num} iteration is {cost}')
        return self.w, self.b

    def plot_cost(self):
        # Plot cost curve
        plt.plot(range(self.iter_num), self.cost_list)
        plt.xlabel('Iteration Number')
        plt.ylabel('Cost')
        plt.show()

    def predict(self, X, y, threshold=0.5):
        # Generate prediction
        y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
        y_label = y_pred > threshold
        print(f'Accuracy is {np.sum(y_label==y)/y.shape[0]}')

        return y_label
