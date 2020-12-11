import numpy as np


# Latent Factor Model
class CF:
    def __init__(self, x, k=2, max_iter=100, learning_rate=0.001, lambd=0.004):
        self.m, self.n = x.shape
        self.x = x
        self.K = k
        self.alpha = learning_rate
        self.lambd = lambd
        self.p = np.random.rand(self.m, k)
        self.q = np.random.rand(self.n, k)
        self.q = self.q.T
        self.max_iter = max_iter
        self.cost = 0

    def train(self):
        for _ in range(self.max_iter):
            self.cost = 0
            for u in range(self.m):
                for i in range(self.n):
                    if self.x[u][i] > 0:
                        error_ui = np.dot(self.p[u, :], self.q[:, i]) - self.x[u][i]
                        self.cost += error_ui ** 2
                        for k in range(self.K):
                            self.p[u][k] = self.p[u][k] - \
                                           self.alpha * (2 * error_ui * self.q[k][i] + 2 * self.lambd * self.p[u][k])
                            self.q[k][i] = self.q[k][i] - \
                                           self.alpha * (2 * error_ui * self.p[u][k] + 2 * self.lambd * self.q[k][i])
                            self.cost += self.lambd * (self.p[u][k] ** 2 + self.q[k][i] ** 2)
        return self.cost

    def predict(self):
        return self.p.dot(self.q)


if __name__ == '__main__':
    data = np.array([[4, 0, 2, 0, 1],
                     [0, 2, 3, 0, 0],
                     [1, 0, 2, 4, 0],
                     [5, 0, 0, 3, 1],
                     [0, 0, 1, 5, 1],
                     [0, 3, 2, 4, 1]])
    cf = CF(data, k=3, max_iter=1000, learning_rate=0.001, lambd=0.004)
    cost = cf.train()
    pred_data = cf.predict()
    print(f'Cost: {cost}')
    print(f'Predicted data:\n{pred_data}')
    recommend_data = pred_data - data
    recommend_idx = [np.argmax(recommend_data[i, :]) for i in range(pred_data.shape[0])]
    user_data = {}
    for i in range(len(recommend_idx)):
        user_data['User' + str(i)] = recommend_idx[i]
    print(user_data)

