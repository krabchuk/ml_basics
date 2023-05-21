from linear_models import LinearModel, LinearModelWithAdditionalElement
from utils import split

import unittest
import numpy as np
import sklearn.metrics
import sklearn.linear_model


np.random.seed(42)

class LinearModelVsSklearnTest(unittest.TestCase):
    def test_training(self):
        data = np.random.rand(100, 3)
        target = 10 * data[:, 0] + 20 * data[:, 1] + 30 * data[:, 2] + np.random.normal(size=100)
        train_data, train_target, test_data, test_target = split(data, target, 0.8)

        model = LinearModel()
        model.fit(train_data, train_target, lr=0.0001, iterations=100000)
        roc_auc = sklearn.metrics.mean_squared_error(test_target, model.predict(test_data))

        val_model = sklearn.linear_model.LinearRegression()
        val_model.fit(train_data, train_target)
        val_roc_auc = sklearn.metrics.mean_squared_error(test_target, val_model.predict(test_data))

        self.assertAlmostEqual(roc_auc, val_roc_auc, delta=0.05)


class LinearModelWithAdditionalElementVsSklearnTest(unittest.TestCase):
    def test_training(self):
        data = np.random.rand(100, 3)
        target = 10 * data[:, 0] + 20 * data[:, 1] + 30 * data[:, 2] + np.random.normal(size=100)
        train_data, train_target, test_data, test_target = split(data, target, 0.8)

        model = LinearModelWithAdditionalElement()
        model.fit(train_data, train_target, lr=0.0001, iterations=100000)
        roc_auc = sklearn.metrics.mean_squared_error(test_target, model.predict(test_data))

        val_model = sklearn.linear_model.LinearRegression()
        val_model.fit(train_data, train_target)
        val_roc_auc = sklearn.metrics.mean_squared_error(test_target, val_model.predict(test_data))

        self.assertAlmostEqual(roc_auc, val_roc_auc, delta=0.01)
