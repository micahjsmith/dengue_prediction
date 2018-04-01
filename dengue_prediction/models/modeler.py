import logging
import sys
import traceback
from collections import defaultdict

import funcy
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dengue_prediction.config import load_config
from dengue_prediction.constants import ProblemType
from dengue_prediction.models import constants
from dengue_prediction.models.constants import (
    ClassificationMetricAgg, MetricComputationApproach)
from dengue_prediction.models.input_type_transforms import (
    FeatureTypeTransformer, TargetTypeTransformer)
from dengue_prediction.models.metrics import Metric, MetricList
from dengue_prediction.util import RANDOM_STATE, str_to_enum_member

logger = logging.getLogger(__name__)

def create_model():
    config = load_config()
    problem_type_str = funcy.get_in(config, ['problem', 'problem_type'])
    problem_type = str_to_enum_member(problem_type_str, ProblemType)
    if problem_type is not None:
        return Modeler(problem_type)
    else:
        # TODO
        raise RuntimeError(
            'Bad problem type in config.yml: {}'.format(problem_type_str))


class Modeler:
    """Versatile modeling object.
    Handles classification and regression problems and computes variety of
    performance metrics.
    Parameters
    ----------
    problem_type : ProblemType
    """

    def __init__(self, problem_type):
        self.problem_type = problem_type
        self.estimator = self._get_default_estimator()
        self.feature_type_transformer = FeatureTypeTransformer()
        self.target_type_transformer = TargetTypeTransformer()

    def set_estimator(self, estimator):
        self.estimator = estimator

    def compute_metrics_cv(self, X, y, **kwargs):
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

    def _compute_metrics_train_test_fitted(self, X, y, classes=None):
        scorings, scorings_ = self._get_scorings()

        # Determine binary/multiclass classification
        if classes is None:
            classes = np.unique(y)
        params = self._get_params(classes)

        scores = {}
        for scoring in scorings_:
            scores[scoring] = self._do_scoring(scoring, params, self.estimator,
                                               X, y)

        metric_list = self.scores_to_metriclist(scorings, scores)
        return metric_list

    def compute_metrics_train_test(self, X, y, n):
        """Compute metrics on test set.
        """

        X, y = self._format_inputs(X, y)

        X_tr, y_tr = X[:n], y[:n]
        X_te, y_te = X[n:], y[n:]

        # fit model on entire training set
        self.estimator.fit(X_tr, y_tr)

        return self._compute_metrics_train_test_fitted(
            X_te, y_te, classes=np.unique(y))

    def _do_scoring(self, scoring, params, model, X_te, y_te,
                    failure_value=None):
        # Make and evaluate predictions. Note that ROC AUC may raise
        # exception if somehow we only have examples from one class in
        # a given fold.
        y_te_transformed = params[scoring]["pred_transformer"](y_te)
        y_te_pred = params[scoring]["predictor"](model, X_te)

        try:
            score = params[scoring]["scorer"](y_te_transformed, y_te_pred)
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

        X, y = self._format_inputs(X, y)

        scorings = list(scorings)

        # Determine binary/multiclass classification
        classes = np.unique(y)
        params = self._get_params(classes)

        if self._is_classification():
            kf = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE + 3)
        elif self._is_regression():
            kf = KFold(shuffle=True, random_state=RANDOM_STATE + 4)
        else:
            raise NotImplementedError

        # Split data, train model, and evaluate metric. We fit the model just
        # once per fold.
        scoring_outputs = defaultdict(lambda: [])
        for inds_tr, inds_te in kf.split(X, y):
            X_tr, X_te = X[inds_tr], X[inds_te]
            y_tr, y_te = y[inds_tr], y[inds_te]

            self.estimator.fit(X_tr, y_tr)

            for scoring in scorings:
                score = self._do_scoring(scoring, params, self.estimator, X_te,
                                         y_te, failure_value=np.nan)
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
            metric_aggregation = (
                ClassificationMetricAgg.BINARY_METRIC_AGGREGATION)
        else:
            metric_aggregation = (
                ClassificationMetricAgg.MULTICLASS_METRIC_AGGREGATION)
        metric_aggregation = metric_aggregation.value

        # Determine predictor (labels, label probabilities, or values) and
        # scoring function.

        # predictors
        def predict(model, X_te):
            return model.predict(X_te)

        def predict_prob(model, X_te):
            return model.predict_proba(X_te)

        # transformers
        def transformer_binarize(y_true):
            return label_binarize(y_true, classes=classes)

        # scorers
        # nothing here

        params = {
            "accuracy": {
                "predictor": predict,
                "pred_transformer": funcy.identity,
                "scorer": sklearn.metrics.accuracy_score,
            },
            "precision": {
                "predictor": predict,
                "pred_transformer": funcy.identity,
                "scorer": lambda y_true, y_pred:
                    sklearn.metrics.precision_score(y_true, y_pred,
                                                    average=metric_aggregation)
            },
            "recall": {
                "predictor": predict,
                "pred_transformer": funcy.identity,
                "scorer": lambda y_true, y_pred:
                    sklearn.metrics.recall_score(y_true, y_pred,
                                                 average=metric_aggregation),
            },
            "roc_auc": {
                "predictor": predict if is_binary else predict_prob,
                "pred_transformer":
                    funcy.identity if is_binary else transformer_binarize,
                "scorer": lambda y_true, y_pred:
                    sklearn.metrics.roc_auc_score(y_true, y_pred,
                                                  average=metric_aggregation),
            },
            "root_mean_squared_error": {
                "predictor": predict,
                "pred_transformer": funcy.identity,
                "scorer": lambda y_true, y_pred:
                    np.sqrt(sklearn.metrics.mean_squared_error(y_true,
                                                               y_pred)),
            },
            "r2": {
                "predictor": predict,
                "pred_transformer": funcy.identity,
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

    def _format_inputs(self, X, y):
        X = self.feature_type_transformer.fit_transform(X)
        y = self.target_type_transformer.fit_transform(y)
        return X, y

    def _get_default_estimator(self):
        if self._is_classification():
            return self._get_default_classifier()
        elif self._is_regression():
            return self._get_default_regressor()
        else:
            raise NotImplementedError

    def _get_default_classifier(self):
        return RandomForestClassifier(random_state=RANDOM_STATE+1)

    def _get_default_regressor(self):
        return RandomForestRegressor(random_state=RANDOM_STATE+2)


class DecisionTreeModeler(Modeler):

    def _get_default_classifier(self):
        return DecisionTreeClassifier(random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return DecisionTreeRegressor(random_state=RANDOM_STATE + 2)
