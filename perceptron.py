import numpy as np


class Perceptron:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                print(f'prediction {prediction}, target {target}')
                print(f'diff, {(target - prediction)}')
                update = self.eta * (target - prediction)
                print(update)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        "calculates base input"
        return X.dot(self.w_[1:]) + self.w_[0]

    def predict(self, X):
        "returs metka class after 1 step"
        return np.where(self.net_input(X) >= 0.0, 1, -1)

