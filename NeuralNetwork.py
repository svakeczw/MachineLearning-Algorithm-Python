import numpy as np
from matplotlib import pyplot as plt


class NN:
    def __init__(self, x, y, layer_dimension, learning_rate):
        """
        :param x: training data x
        :param y: training data y
        :param layer_dimension: dimension of network ie: [num, num...]
        :param learning_rate: learning rate alpha
        """
        self.n, self.m = x.shape
        self.layer_dim = [self.n] + layer_dimension
        self.y = y
        self.alpha = learning_rate
        self.L = len(self.layer_dim)
        self.parameters = {}
        self.grad = {}

    def ini_parameters(self):
        """
        initialize parameters (two methods)
        :return: parameters dic(w,b)
        """
        for i in range(1, self.L):
            # self.parameters['w' + str(i)] = np.random.randn(self.layer_dim[i],self.layer_dim[i-1]) * 0.01  # method 1
            self.parameters['w' + str(i)] = np.random.randn(self.layer_dim[i], self.layer_dim[i-1])\
                                            * np.sqrt(2.0 / self.layer_dim[i-1])  # method 2
            self.parameters['b' + str(i)] = np.zeros((self.layer_dim[i], 1))
            assert self.parameters['w' + str(i)].shape == (self.layer_dim[i], self.layer_dim[i-1])
            assert self.parameters['b' + str(i)].shape == (self.layer_dim[i], 1)

        return self.parameters

    @staticmethod
    def sigmoid(z):
        # Sigmoid function
        A = 1 / (1 + np.exp(-z))
        cache = z  # activation cache
        assert A.shape == z.shape

        return A, cache

    @staticmethod
    def relu(z):
        # ReLU function
        A = np.maximum(0, z)
        cache = z  # activation cache
        assert A.shape == z.shape
        return A, cache

    @staticmethod
    def linear_forward(A, W, b):
        """
        :param A: activation
        :param W: weights
        :param b: bias
        :return: z, linear cache
        """
        # print(f'W:{W.shape}')
        # print(f'A:{A.shape}')
        z = W.dot(A) + b
        cache = (A, W, b)  # linear cache
        assert z.shape == (W.shape[0], A.shape[1])
        return z, cache

    def activation_forward(self, pre_A, W, b, method):
        """
        :param pre_A: activation from previous layer
        :param W: weights
        :param b: bias
        :param method: Sigmoid/ReLU
        :return: cache set of linear cache and activation cache
        """
        if method == 'sigmoid':
            # use a[l-1] to compute a[l]
            z, linear_cache = self.linear_forward(pre_A, W, b)
            A, activation_cache = self.sigmoid(z)
        if method == 'relu':
            # use a[l-1] to compute a[l]
            z, linear_cache = self.linear_forward(pre_A, W, b)
            A, activation_cache = self.relu(z)

        assert A.shape == (W.shape[0], pre_A.shape[1])

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self, x, parameters):
        """
        forward propagation
        :param x: data x
        :param parameters: parameters dic
        :return: activation of the last layer, cache list of cache sets
        """
        A = x  # A0 = x
        cache_list = []
        L = len(parameters) // 2

        # forward propagation through hidden layers
        for l in range(1, L):
            pre_A = A
            A, cache = self.activation_forward(pre_A, parameters['w' + str(l)], parameters['b' + str(l)], method='relu')
            cache_list.append(cache)  # (linear_cache, activation_cache)

        # forward propagation through last layer
        final_A, cache = self.activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], method='sigmoid')
        cache_list.append(cache)
        assert final_A.shape == (1, x.shape[1])  # (1, m)

        return final_A, cache_list

    def cost_function(self, final_A, y):
        """
        compute cost
        :param final_A: activation of the last layer
        :param y: y data
        :return: cost
        """
        # binary classification loss
        cost = (-1/self.m) * np.sum(y*np.log(final_A) + (1-y)*np.log(1-final_A), axis=1, keepdims=True)
        cost = np.squeeze(cost)
        assert cost.shape == ()

        return cost

    @staticmethod
    def sigmoid_back(dA, activation_cache):
        """
        :param dA: d(loss)/d(a)
        :param activation_cache: z
        :return: d(loss)/d(z)
        """
        z = activation_cache
        s = 1/(1+np.exp(-z))
        dz = dA * s * (1-s)  # d(loss)/d(z) = d(loss)/d(a) * d(a)/d(z)
        assert dz.shape == z.shape

        return dz

    @staticmethod
    def relu_back(dA, activation_cache):
        """
        :param dA: d(loss)/d(a)
        :param activation_cache: z
        :return: d(loss)/d(z)
        """
        z = activation_cache
        dz = np.array(dA)
        dz[z <= 0] = 0
        return dz

    @staticmethod
    def linear_back(dz, linear_cache):
        """
        :param dz: d(loss)/d(z)
        :param linear_cache: previous layer activation, weights, bias
        :return: d(loss)/d(a_previous_layer), d(loss)/d(weights), d(loss)/d(bias)
        """
        pre_A, W, b = linear_cache
        m = pre_A.shape[1]

        dw = (1/m) * dz.dot(pre_A.T)
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        dpre_A = np.dot(W.T, dz)

        assert dw.shape == W.shape
        assert db.shape == b.shape
        assert dpre_A.shape == pre_A.shape

        return dpre_A, dw, db

    def activation_back(self, dA, cache_list, method):
        """
        :param dA: d(loss).d(activation)
        :param cache_list: cache set of linear cache and activation cache
        :param method: Sigmoid/ReLU
        :return: d(loss)/d(activation_previous_layer), d(loss)/d(weights), d(loss)/d(bias)
        """
        linear_cache, activation_cache = cache_list

        if method == 'sigmoid':
            dz = self.sigmoid_back(dA, activation_cache)
            dpre_A, dw, db = self.linear_back(dz, linear_cache)
        if method == 'relu':
            dz = self.relu_back(dA, activation_cache)
            dpre_A, dw, db = self.linear_back(dz, linear_cache)
        return dpre_A, dw, db

    def back_prop(self, final_A, cache_list):
        """
        back propagation
        :param final_A: activation of the last layer
        :param cache_list: cache list of cache sets
        :return: gradient
        """
        # print(f'final_A: {final_A.shape}')
        assert final_A.shape == self.y.shape
        L = len(cache_list)

        # back propagation though last layer
        dfinal_A = - (np.divide(self.y, final_A) - np.divide(1-self.y, 1-final_A))

        grad_cache = self.activation_back(dfinal_A, cache_list[L-1], 'sigmoid')
        self.grad['dA' + str(L-1)] = grad_cache[0]
        self.grad['dw' + str(L)] = grad_cache[1]
        self.grad['db' + str(L)] = grad_cache[2]

        # back propagation though the last layer(L-1) to layer 1(l+1)
        for l in reversed(range(L-1)):
            grad_cache = self.activation_back(self.grad['dA' + str(l+1)], cache_list[l], 'relu')
            self.grad['dA' + str(l)] = grad_cache[0]
            self.grad['dw' + str(l+1)] = grad_cache[1]
            self.grad['db' + str(l+1)] = grad_cache[2]
        return self.grad

    def parameter_update(self):
        """
        :return: updated parameters(weights, bias)
        """
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters['w' + str(l+1)] = self.parameters['w' + str(l+1)] - self.alpha * self.grad['dw' + str(l+1)]
            self.parameters['b' + str(l+1)] = self.parameters['b' + str(l+1)] - self.alpha * self.grad['db' + str(l+1)]
        return self.parameters

    def train(self, x, y, iter_num=100):
        """
        :param x: data x
        :param y: data y
        :param iter_num: number of iterations
        :return: parameters(weights, bias)
        """
        self.parameters = self.ini_parameters()
        costs = []

        for i in range(iter_num):
            final_A, cache_list = self.forward_prop(x, self.parameters)
            cost = self.cost_function(final_A, y)
            costs.append(cost)
            self.grad = self.back_prop(final_A, cache_list)
            self.parameters = self.parameter_update()
            if i % 100 == 0 and i != 0:
                # print cost every 100 iterations
                print(f'iteration: {i} cost: {cost}')

        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title(f'Learning rate = {self.alpha}')
        plt.show()
        return self.parameters

    def predict(self, x, y):
        """
        predict labels
        :param x: test x
        :param y: test y
        :return:
        """
        n, m = x.shape
        y_pred, _ = self.forward_prop(x, self.parameters)
        for i in range(0, y_pred.shape[1]):
            if y_pred[0, i] > 0.5:
                y_pred[0, i] = 1
            else:
                y_pred[0, i] = 0
        print(f'Accuracy: {np.sum(y_pred==y)/m}')
