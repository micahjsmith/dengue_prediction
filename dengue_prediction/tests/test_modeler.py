import unittest

import pandas as pd
import sklearn.datasets

from dengue_prediction.constants import ProblemType
from dengue_prediction.models.modeler import (
    DecisionTreeModeler, TunedModeler, TunedRandomForestClassifier,
    TunedRandomForestRegressor)
from dengue_prediction.tests.util import EPSILON


class _CommonTesting:

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

    def _test_problem_type_cv(self, problem_type, data):
        model = self.ModelerClass(problem_type)
        X = data[problem_type]["X"]
        y = data[problem_type]["y"]
        metrics = model.compute_metrics_cv(X, y)

        return metrics

    def _test_problem_type_train_test(self, problem_type, data):
        model = self.ModelerClass(problem_type)
        X = data[problem_type]["X"]
        y = data[problem_type]["y"]
        n = round(0.7 * len(X))
        metrics = model.compute_metrics_train_test(X, y, n=n)

        return metrics

    def _call_method(self, method, problem_type):
        metrics = getattr(self, method)(problem_type, self.data)
        metrics_pd = getattr(self, method)(problem_type, self.data_pd)
        return metrics, metrics_pd

class TestModeler(_CommonTesting, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.ModelerClass = DecisionTreeModeler

    def test_classification_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', ProblemType.CLASSIFICATION)
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
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', ProblemType.CLASSIFICATION)
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

    def test_regression_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', ProblemType.REGRESSION)
        self.assertEqual(metrics, metrics_pd)
        metrics_user = metrics.convert(kind="user")
        self.assertAlmostEqual(
            metrics_user['Root Mean Squared Error'], 4.4761438, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['R-squared'], 0.7393219, delta=EPSILON)

    def test_regression_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', ProblemType.REGRESSION)
        self.assertEqual(metrics, metrics_pd)
        metrics_user = metrics.convert(kind="user")
        self.assertAlmostEqual(
            metrics_user['Root Mean Squared Error'], 6.9803059, delta=EPSILON)
        self.assertAlmostEqual(
            metrics_user['R-squared'], 0.2656004, delta=EPSILON)


class TestTunedModelers(_CommonTesting, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.ModelerClass = TunedModeler

    def _test_tuned_random_forest_estimator(self, Estimator, problem_type):
        model = Estimator()
        data = self.data[problem_type]
        X, y = data['X'], data['y']
        model.fit(X, y, tune=False)
        old_score = model.score(X, y)
        model.fit(X, y, tune=True)
        new_score = model.score(X, y)
        self.assertGreaterEqual(new_score, old_score)

    def test_tuned_random_forest_regressor(self):
        self._test_tuned_random_forest_estimator(TunedRandomForestRegressor, ProblemType.REGRESSION)

    def test_tuned_random_forest_classifier(self):
        self._test_tuned_random_forest_estimator(TunedRandomForestClassifier, ProblemType.CLASSIFICATION)

    def test_classification_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', ProblemType.CLASSIFICATION)
        self.assertEqual(metrics, metrics_pd)
    def test_classification_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', ProblemType.CLASSIFICATION)
        self.assertEqual(metrics, metrics_pd)
    def test_regression_cv(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_cv', ProblemType.REGRESSION)
    def test_regression_train_test(self):
        metrics, metrics_pd = self._call_method(
            '_test_problem_type_train_test', ProblemType.REGRESSION)
        self.assertEqual(metrics, metrics_pd)
