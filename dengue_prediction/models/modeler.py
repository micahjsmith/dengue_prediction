import importlib
import logging
import os

import funcy
import numpy as np
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate, train_test_split)
from sklearn.model_selection._validation import _multimetric_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dengue_prediction.config import load_config
from dengue_prediction.constants import ProblemTypes
from dengue_prediction.exceptions import ConfigurationError
from dengue_prediction.models import constants
from dengue_prediction.models.input_type_transforms import (
    FeatureTypeTransformer, TargetTypeTransformer)
from dengue_prediction.util import RANDOM_STATE, str_to_enum_member

try:
    import btb
    import btb.tuning.gp
except ImportError:
    btb = None

logger = logging.getLogger(__name__)


def create_model(tuned=True):
    if tuned:
        ModelerClass = TunedModeler
    else:
        ModelerClass = Modeler
    config = load_config()
    problem_type_str = funcy.get_in(config, ['problem', 'problem_type'])
    problem_type = str_to_enum_member(problem_type_str, ProblemType)
    if problem_type is not None:
        logger.info('Initializing {} modeler...'.format(ModelerClass.__name__))
        modeler = ModelerClass(problem_type)
        logger.info(
            'Initializing {} modeler...DONE'.format(ModelerClass.__name__))
        return modeler
    else:
        # TODO
        raise RuntimeError(
            'Bad problem type in config.yml: {}'.format(problem_type_str))


def get_scorer_from_config():
    config = load_config()
    scorer = funcy.get_in(
        config, ['problem', 'problem_type_details', 'scorer'])
    return get_scorer(scorer)


def get_scorer(scorer_name):
    found = False
    try:
        scoring = sklearn.metrics.get_scorer(scorer_name)
        found = True
    except ValueError:
        pass

    if not found:
        i = scorer_name.rfind('.')
        if i < 0:
            raise ValueError(
                'Invalid scorer import path: {}'.format(scorer_name))
        module_name, scorer_name_ = scorer_name[:i], scorer_name[i + 1:]
        mod = importlib.import_module(module_name)
        scoring = getattr(mod, scorer_name_)
        found = True

    if not found:
        raise ConfigurationError(
            'Could not get a scorer with configuration {}'.format(scorer_name))

    return scoring


def scoring_name_to_name(scoring_name):
    mapper = constants.SCORING_NAME_MAPPER

    if scoring_name in mapper:
        return mapper[scoring_name]
    else:
        # default formatting
        def upper_first(s):
            if not s:
                return s
            elif len(s) == 1:
                return s.upper()
            else:
                return s[0].upper() + s[1:]
        format = funcy.rcompose(
            lambda s: s.split('_'),
            funcy.partial(map, upper_first),
            lambda l: ' '.join(l),
        )
        return format(scoring_name)


def name_to_scoring_name(name):
    mapper = funcy.flip(constants.SCORING_NAME_MAPPER)
    if name in mapper:
        return mapper[name]
    else:
        # default formatting
        return '_'.join(name.lower().split(' '))


class Modeler:
    """Versatile modeling object.
    Handles classification and regression problems and computes variety of
    performance metrics.
    Parameters
    ----------
    problem_type : ProblemType
    """

    def __init__(self,
                 problem_type=None,
                 scorer=None,
                 classification_type=None,
                 ):
        self.problem_type = problem_type
        self.scorer = scorer
        self.classification_type = classification_type

        self.estimator = self._get_default_estimator()
        self.feature_type_transformer = FeatureTypeTransformer()
        self.target_type_transformer = TargetTypeTransformer()

    @classmethod
    def from_config(cls):
        config = load_config()
        c = funcy.partial(funcy.get_in, config)
        problem_type = c(['problem', 'problem_type'])
        scorer = c(['problem', 'problem_type_details', 'scorer'])

        if problem_type == 'classification':
            classification_type = c(
                ['problem', 'problem_type_details', 'classification_type'])
            # TODO more?

        return cls(problem_type=problem_type,
                   scorer=scorer,
                   classification_type=classification_type)

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

        scoring_names = self._get_scoring_names()

        # compute scores
        results = self.cv_score_mean(X, y, scoring_names)
        return results

    def fit(self, X, y, **kwargs):
        X, y = self._format_inputs(X, y)
        self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        X = self._format_X(X)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        X = self._format_X(X)
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        X, y = self._format_inputs(X, y)
        return self.estimator.score(X, y)

    def dump(self, filepath):
        joblib.dump(self.estimator, filepath, compress=True)

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise ValueError("Couldn't find model at {}".format(filepath))
        self.estimator = joblib.load(filepath)

    def compute_metrics_train_test(self, X, y, n):
        """Compute metrics on test set.
        """

        X, y = self._format_inputs(X, y)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=n, test_size=len(y) - n, shuffle=True)

        # fit model on entire training set
        self.estimator.fit(X_tr, y_tr)

        scoring_names = self._get_scoring_names()
        scorers = {
            s: sklearn.metrics.get_scorer(s)
            for s in scoring_names
        }
        multimetric_score_results = _multimetric_score(
            self.estimator, X_te, y_te, scorers)

        results = self._process_cv_results(
            multimetric_score_results, filter_testing_keys=False)
        return results

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

        if self._is_classification():
            kf = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE + 3)
        elif self._is_regression():
            kf = KFold(shuffle=True, random_state=RANDOM_STATE + 4)
        else:
            raise NotImplementedError

        cv_results = cross_validate(
            self.estimator, X, y,
            scoring=scorings, cv=kf, return_train_score=False)

        # post-processing
        results = self._process_cv_results(cv_results)
        return results

    def _process_cv_results(self, cv_results, filter_testing_keys=True):
        result = []
        for key, val in cv_results.items():
            if filter_testing_keys:
                if key.startswith('test_'):
                    scoring_name = key[len('test_'):]
                else:
                    continue
            else:
                scoring_name = key
            name = scoring_name_to_name(scoring_name)
            val = np.nanmean(cv_results[key])
            if np.isnan(val):
                val = None
            result.append({
                'name': name,
                'scoring_name': scoring_name,
                'value': val,
            })

        return result

    def _is_classification(self):
        return self.problem_type == ProblemTypes.CLASSIFICATION

    def _is_regression(self):
        return self.problem_type == ProblemTypes.REGRESSION

    def _get_scoring_names(self):
        """Get scorings for this problem type.
        Returns
        -------
        scoring_names: list
            List of "scoring" as defined in sklearn.metrics. This is a "utility
            variable" that can be used where we just need the names of the
            scoring functions and not the more complete information.
        """
        # scoring_types maps user-readable name to `scoring`, as argument to
        # cross_val_score
        # See also
        # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        if self._is_classification():
            return constants.CLASSIFICATION_SCORING
        elif self._is_regression():
            return constants.REGRESSION_SCORING
        else:
            raise NotImplementedError

    def _format_inputs(self, X, y):
        return self._format_X(X), self._format_y(y)

    def _format_y(self, y):
        return self.target_type_transformer.fit_transform(y)

    def _format_X(self, X):
        return self.feature_type_transformer.fit_transform(X)

    def _get_default_estimator(self):
        if self._is_classification():
            return self._get_default_classifier()
        elif self._is_regression():
            return self._get_default_regressor()
        else:
            raise NotImplementedError

    def _get_default_classifier(self):
        return RandomForestClassifier(random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return RandomForestRegressor(random_state=RANDOM_STATE + 2)


class DecisionTreeModeler(Modeler):

    def _get_default_classifier(self):
        return DecisionTreeClassifier(random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return DecisionTreeRegressor(random_state=RANDOM_STATE + 2)


class SelfTuningMixin:

    # overwrite this to do anything
    def get_tunables(self):
        return None

    @property
    def tunables(self):
        if not hasattr(self, '_tunables'):
            self._tunables = self.get_tunables()
        return self._tunables

    @tunables.setter
    def tunables(self, tunables):
        self._tunables = tunables

    @property
    def tuning_cv(self):
        if not hasattr(self, '_tuning_cv'):
            self._tuning_cv = 3
        return self._tuning_cv

    @tuning_cv.setter
    def tuning_cv(self, tuning_cv):
        self._tuning_cv = tuning_cv

    @property
    def tuning_iter(self):
        if not hasattr(self, '_tuning_iter'):
            self._tuning_iter = 3
        return self._tuning_iter

    @tuning_iter.setter
    def tuning_iter(self, tuning_iter):
        self._tuning_iter = tuning_iter

    def _get_parent_instance(self):
        # this is probably a sign of bad design pattern
        mro = type(self).__mro__
        ParentClass = mro[mro.index(__class__) + 1]  # noqa
        return ParentClass()

    def fit(self, X, y, tune=True, **fit_kwargs):
        if tune:
            # do some tuning
            if btb is not None and self.tunables is not None:

                # make scoring driver using scorer as specified in config
                scorer = get_scorer_from_config()

                def score(estimator):
                    scores = cross_val_score(
                        estimator, X, y,
                        scoring=scorer, cv=self.tuning_cv,
                        fit_params=fit_kwargs)
                    return np.mean(scores)

                logger.info('Tuning model using BTB GP tuner...')
                tuner = btb.tuning.gp.GP(self.tunables)
                estimator = self._get_parent_instance()
                original_score = score(estimator)
                for i in range(self.tuning_iter):
                    params = tuner.propose()
                    estimator.set_params(**params)
                    score_ = score(estimator)
                    tuner.add(params, score_)

                best_params = tuner._best_hyperparams
                best_score = tuner._best_score
                self.set_params(**best_params)
                logger.info(
                    'Tuning complete. '
                    'Cross val score changed from {0:.3f} to {0:.3f}.'
                    .format(original_score, best_score))
            else:
                logging.warning('Tuning requested, but either btb not '
                                'installed or tunable HyperParameters not '
                                'specified.')

        return super().fit(X, y, **fit_kwargs)


class SelfTuningRandomForestMixin(SelfTuningMixin):
    def get_tunables(self):
        if btb is not None:
            return [
                ('n_estimators',
                 btb.HyperParameter(btb.ParamTypes.INT, [10, 500])),
                ('max_depth',
                 btb.HyperParameter(btb.ParamTypes.INT, [3, 20]))
            ]
        else:
            return None


class TunedRandomForestRegressor(
        SelfTuningRandomForestMixin, RandomForestRegressor):
    pass


class TunedRandomForestClassifier(
        SelfTuningRandomForestMixin, RandomForestClassifier):
    pass


class TunedModeler(Modeler):
    def _get_default_classifier(self):
        return TunedRandomForestClassifier(random_state=RANDOM_STATE + 1)

    def _get_default_regressor(self):
        return TunedRandomForestRegressor(random_state=RANDOM_STATE + 2)
