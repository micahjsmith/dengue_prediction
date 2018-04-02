import unittest

import pandas as pd
import sklearn.datasets

from dengue_prediction.constants import ProblemType
from dengue_prediction.models.modeler import DecisionTreeModeler
from dengue_prediction.tests.util import EPSILON


class TestModeler(unittest.TestCase):
    def setUp(self):
        # Create fake data
        X_classification, y_classification = sklearn.datasets.load_iris(
            return_X_y=True)
        X_regression, y_regression = sklearn.datasets.load_boston(
            return_X_y=True)

        self.data = {
            ProblemType.CLASSIFICATION: {
                "X": X_classification,
                "y": y_classification,
            },
            ProblemType.REGRESSION: {
                "X": X_regression,
                "y": y_regression,
            },
        }

        self.data_pd = {
            ProblemType.CLASSIFICATION: {
                "X": pd.DataFrame(X_classification),
                "y": pd.DataFrame(y_classification),
            },
            ProblemType.REGRESSION: {
                "X": pd.DataFrame(X_regression),
                "y": pd.DataFrame(y_regression),
            },
        }

    def test_classification(self):
        metrics = self._test_problem_type_cv(
            ProblemType.CLASSIFICATION, self.data)
        metrics_pd = self._test_problem_type_cv(
            ProblemType.CLASSIFICATION, self.data_pd)

        self.assertEqual(metrics, metrics_pd)

        metrics_user = metrics.convert(kind="user")

        self.assertAlmostEqual(
            metrics_user['Accuracy'], 0.9403594, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['Precision'], 0.9403594, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['Recall'], 0.9403594, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['ROC AUC'], 0.9552696, delta=EPSILON)

    def test_classification_train_test(self):
        metrics = self._test_problem_type_train_test(
            ProblemType.CLASSIFICATION, self.data)
        metrics_pd = self._test_problem_type_train_test(
            ProblemType.CLASSIFICATION, self.data_pd)

        self.assertEqual(metrics, metrics_pd)

        metrics_user = metrics.convert(kind="user")
        self.assertAlmostEqual(
            metrics_user['Accuracy'], 0.7777777, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['Precision'], 0.7777777, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['Recall'], 0.7777777, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['ROC AUC'], 0.8333333, delta=EPSILON)

    def test_regression(self):
        metrics = self._test_problem_type_cv(ProblemType.REGRESSION, self.data)
        metrics_pd = self._test_problem_type_cv(
            ProblemType.REGRESSION, self.data_pd)

        self.assertEqual(metrics, metrics_pd)

        metrics_user = metrics.convert(kind="user")
        self.assertAlmostEqual(
            metrics_user['Root Mean Squared Error'], 4.4761438, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['R-squared'], 0.7393219, delta=EPSILON)

    def test_regression_train_test(self):
        metrics = self._test_problem_type_train_test(
            ProblemType.REGRESSION, self.data)
        metrics_pd = self._test_problem_type_train_test(
            ProblemType.REGRESSION, self.data_pd)

        self.assertEqual(metrics, metrics_pd)

        metrics_user = metrics.convert(kind="user")
        self.assertAlmostEqual(
            metrics_user['Root Mean Squared Error'], 6.9803059, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['R-squared'], 0.2656004, delta=EPSILON)

    def _test_problem_type_cv(self, problem_type, data):
        model = DecisionTreeModeler(problem_type)
        X = data[problem_type]["X"]
        y = data[problem_type]["y"]
        metrics = model.compute_metrics_cv(X, y)

        return metrics

    def _test_problem_type_train_test(self, problem_type, data):
        model = DecisionTreeModeler(problem_type)
        X = data[problem_type]["X"]
        y = data[problem_type]["y"]
        n = round(0.7 * len(X))
        metrics = model.compute_metrics_train_test(X, y, n=n)

        return metrics
