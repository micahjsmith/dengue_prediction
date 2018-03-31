import sys
import traceback
from collections import defaultdict

import numpy as np
import sklearn.metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dengue_prediction.models import constants
from dengue_prediction.models.constants import (ClassificationMetricAggregation,
                                                MetricComputationApproach,
                                                ProblemType)
from dengue_prediction.models.metrics import Metric, MetricList
from dengue_prediction.util import RANDOM_STATE


class Modeler:
    """Versatile modeling object.
    Handles classification and regression problems and computes variety of
    performance metrics.
    Parameters
    ----------
    problem_type : str
        One of "classification" or "regression"
    """

    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.model = self._get_default_estimator()

    def compute_metrics(
            self, X, y, kind=MetricComputationApproach.CV, **kwargs):
        if kind == MetricComputationApproach.CV:
            return self.compute_metrics_cv(X, y, **kwargs)
        elif kind == MetricComputationApproach.TRAIN_TEST:
            return self.compute_metrics_train_test(X, y, **kwargs)
        else:
            raise ValueError("Bad metrics kind: {}".format(kind))

    def compute_metrics_cv(self, X, y):
        """Compute cross-validated metrics.
        Trains this model on data X with labels y.
        Returns a MetricList with the name, scoring type, and value for each
        Metric. Note that these values may be numpy floating points, and should
        be converted prior to insertion in a database.
        Parameters
        ----------
        X : numpy array-like or pd.DataFrame
            data
        y : numpy array-like or pd.DataFrame or pd.DataSeries
            labels
        """

        scorings, scorings_ = self._get_scorings()

        # compute scores
        scores = self.cv_score_mean(X, y, scorings_)

        # unpack into MetricList
        metric_list = self.scores_to_metriclist(scorings, scores)
        return metric_list

    def compute_metrics_train_test(self, X, y, n):
        """Compute metrics on test set.
        """

        X, y = Modeler._format_matrices(X, y)

        X_train, y_train = X[:n], y[:n]
        X_test, y_test = X[n:], y[n:]

        scorings, scorings_ = self._get_scorings()

        # Determine binary/multiclass classification
        classes = np.unique(y)
        params = self._get_params(classes)

        # fit model on entire training set
        self.model.fit(X_train, y_train)

        scores = {}
        for scoring in scorings_:
            scores[scoring] = self._do_scoring(scoring, params, self.model,
                                               X_test, y_test)

        metric_list = self.scores_to_metriclist(scorings, scores)
        return metric_list

    def _do_scoring(self, scoring, params, model, X_test, y_test,
                    failure_value=None):
        # Make and evaluate predictions. Note that ROC AUC may raise
        # exception if somehow we only have examples from one class in
        # a given fold.
        y_test_transformed = params[scoring]["pred_transformer"](y_test)
        y_test_pred = params[scoring]["predictor"](model, X_test)

        try:
            score = params[scoring]["scorer"](y_test_transformed, y_test_pred)
        except ValueError as e:
            score = failure_value
            print(traceback.format_exc(), file=sys.stderr)
            raise RuntimeError

        return score

    def cv_score_mean(self, X, y, scorings):
        """Compute mean score across cross validation folds.
        Split data and labels into cross validation folds and fit the model for
        each fold. Then, for each scoring type in scorings, compute the score.
        Finally, average the scores across folds. Returns a dictionary mapping
        scoring to score.
        Parameters
        ----------
        X : numpy array-like
            data
        y : numpy array-like
            labels
        scorings : list of str
            scoring types
        """

        X, y = Modeler._format_matrices(X, y)

        scorings = list(scorings)

        # Determine binary/multiclass classification
        classes = np.unique(y)
        params = self._get_params(classes)

        if self._is_classification():
            kf = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE + 3)
        else:
            kf = KFold(shuffle=True, random_state=RANDOM_STATE + 4)

        # Split data, train model, and evaluate metric. We fit the model just
        # once per fold.
        scoring_outputs = defaultdict(lambda: [])
        for train_inds, test_inds in kf.split(X, y):
            X_train, X_test = X[train_inds], X[test_inds]
            y_train, y_test = y[train_inds], y[test_inds]

            self.model.fit(X_train, y_train)

            for scoring in scorings:
                score = self._do_scoring(scoring, params, self.model, X_test,
                                         y_test, failure_value=np.nan)
                scoring_outputs[scoring].append(score)

        for scoring in scoring_outputs:
            score_mean = np.nanmean(scoring_outputs[scoring])
            if np.isnan(score_mean):
                score_mean = None
            scoring_outputs[scoring] = score_mean

        return scoring_outputs

    def scores_to_metriclist(self, scorings, scores):
        metric_list = MetricList()
        for v in scorings:
            name = v["name"]
            scoring = v["scoring"]

            if scoring in scores:
                value = scores[scoring]
            else:
                value = None

            metric_list.append(Metric(name, scoring, value))

        return metric_list

    def _is_classification(self):
        return self.problem_type == ProblemType.CLASSIFICATION

    def _is_regression(self):
        return self.problem_type == ProblemType.REGRESSION

    def _get_params(self, classes):
        n_classes = len(classes)
        is_binary = n_classes == 2
        if is_binary:
            metric_aggregation = ClassificationMetricAggregation.BINARY_METRIC_AGGREGATION
        else:
            metric_aggregation = ClassificationMetricAggregation.MULTICLASS_METRIC_AGGREGATION
        metric_aggregation = metric_aggregation.value

        # Determine predictor (labels, label probabilities, or values) and
        # scoring function.

        # predictors
        def predict(model, X_test):
            return model.predict(X_test)

        def predict_prob(model, X_test):
            return model.predict_proba(X_test)

        # transformers
        def noop(y_true):
            return y_true

        def transformer_binarize(y_true):
            return label_binarize(y_true, classes=classes)

        # scorers
        # nothing here

        params = {
            "accuracy": {
                "predictor": predict,
                "pred_transformer": noop,
                "scorer": sklearn.metrics.accuracy_score,
            },
            "precision": {
                "predictor": predict,
                "pred_transformer": noop,
                "scorer": lambda y_true, y_pred: sklearn.metrics.precision_score(
                    y_true, y_pred, average=metric_aggregation),
            },
            "recall": {
                "predictor": predict,
                "pred_transformer": noop,
                "scorer": lambda y_true, y_pred: sklearn.metrics.recall_score(
                    y_true, y_pred, average=metric_aggregation),
            },
            "roc_auc": {
                "predictor": predict if is_binary else predict_prob,
                "pred_transformer": noop if is_binary else transformer_binarize,
                "scorer": lambda y_true, y_pred: sklearn.metrics.roc_auc_score(
                    y_true, y_pred, average=metric_aggregation),
            },
            "root_mean_squared_error": {
                "predictor": predict,
                "pred_transformer": noop,
                "scorer": lambda y_true, y_pred:
                    np.sqrt(sklearn.metrics.mean_squared_error(y_true,
                                                               y_pred)),
            },
            "r2": {
                "predictor": predict,
                "pred_transformer": noop,
                "scorer": sklearn.metrics.r2_score
            },
        }

        return params

    def _get_scorings(self):
        """Get scorings for this problem type.
        Returns
        -------
        scorings : list of dict
            Information on metric name and associated "scoring" as defined in
            sklearn.metrics
        scorings_ : list
            List of "scoring" as defined in sklearn.metrics. This is a "utility
            variable" that can be used where we just need the names of the
            scoring functions and not the more complete information.
        """
        # scoring_types maps user-readable name to `scoring`, as argument to
        # cross_val_score
        # See also
        # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if self._is_classification():
            scorings = constants.CLASSIFICATION_SCORING
            scorings_ = [s["scoring"] for s in scorings]
        elif self._is_regression():
            scorings = constants.REGRESSION_SCORING
            scorings_ = [s["scoring"] for s in scorings]
        else:
            raise NotImplementedError

        return scorings, scorings_

    @staticmethod
    def _format_matrices(X, y):
        X = Modeler._formatX(X)
        y = Modeler._formaty(y)
        return X, y

    @staticmethod
    def _formatX(X):
        # ensure that we use np for everything
        # use np.float64 for all elements
        # *don't* use 1d array for X
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    @staticmethod
    def _formaty(y):
        # TODO: detect if we need to use a LabelEncoder for y
        # ensure that we use np for everything
        # use np.float64 for all elements
        # *do* use 1d array for y
        y = np.asarray(y)
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError("Target matrix has too many columns: {}"
                             .format(y.shape[1]))
        y = y.ravel()
        return y

    def _get_default_estimator(self):
        if self._is_classification():
            return Modeler._get_default_classifier()
        elif self._is_regression():
            return Modeler._get_default_regressor()
        else:
            raise NotImplementedError

    @staticmethod
    def _get_default_classifier():
        return DecisionTreeClassifier(random_state=RANDOM_STATE + 1)
        # return RandomForestClassifier(random_state=RANDOM_STATE+1)

    @staticmethod
    def _get_default_regressor():
        return DecisionTreeRegressor(random_state=RANDOM_STATE + 2)
        # return RandomForestRegressor(random_state=RANDOM_STATE+2)
