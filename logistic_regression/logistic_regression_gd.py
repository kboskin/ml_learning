import numpy as np


class LogisticRegressionGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        print("shape is", X.shape[1])
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print("w_ is", self.w_)
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        "calculates base input"
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        "returs metka class after 1 step"
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
