import numpy as np

import sys
print(sys.path)


class LinearModel:
    def __init__(self):
        self.weights = None
        self.train_losses = None

    def __make_iteration(self, X, y, lr, min_loss=None, debug_print=False):
        y_pred = np.dot(X, self.weights)
        if debug_print:
            print(self.weights)
        loss = np.sum((y - y_pred) ** 2) / len(y)
        self.train_losses.append(loss)
        grad = np.dot(y_pred - y, X)
        if min_loss is not None and np.sum(np.abs(grad)) < min_loss:
            return
        self.weights -= lr * grad

    def fit(self, X, y, lr=0.01, iterations=1000, min_loss=1e-9, debug_print=False):
        self.weights = np.random.rand(X.shape[1])
        self.train_losses = []
        for i in range(iterations):
            self.__make_iteration(X, y, lr, min_loss, debug_print)

    def predict(self, X):
        return np.dot(X, self.weights)


class LinearModelWithAdditionalElement():
    def __init__(self):
        self.weights = None
        self.train_losses = None

    def __make_iteration(self, X, y, lr, min_loss=None):
        y_pred = np.dot(X, self.weights)
        loss = np.sum((y - y_pred) ** 2) / len(y)
        self.train_losses.append(loss)
        grad = 2 * np.dot(y_pred - y, X)
        if min_loss is not None and np.sum(np.abs(grad)) < min_loss:
            return
        self.weights -= lr * grad

    def fit(self, X, y, lr=0.01, iterations=1000, min_loss=1e-12):
        self.train_losses = []
        _X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
        self.weights = np.random.rand(_X.shape[1])
        for i in range(iterations):
            self.__make_iteration(_X, y, lr, min_loss)

    def predict(self, X):
        _X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
        return np.dot(_X, self.weights)
