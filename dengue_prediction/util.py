import inspect
import logging
import random

import funcy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.pipeline import TransformerPipeline

RANDOM_STATE = 1754


def asarray2d(a):
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_kwargs):
        return self

    def transform(self, X, **transform_kwargs):
        return X


class FragileTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, bad_input_checks, errors):
        '''Raises a random error if any input check returns True'''
        super().__init__()

        self._check = funcy.any_fn(*bad_input_checks)
        self._errors = errors

        self._random = random.Random()
        self._random.seed = hash(funcy.merge(bad_input_checks, errors))

    def _raise(self):
        raise self._random.choice(self._errors)

    def fit(self, X, y=None, **fit_kwargs):
        if self._check(X) or self._check(y):
            self._raise()

        return self

    def transform(self, X, **transform_kwargs):
        if self._check(X):
            self._raise()

        return X


class FragileTransformerPipeline(TransformerPipeline):
    def __init__(self, nsteps, bad_input_checks, errors, shuffle=True, seed=1):
        steps = [
            ('IdentityTransformer{:02d}'.format(i), IdentityTransformer())
            for i in range(nsteps - 1)
        ]
        fragile_transformer = FragileTransformer(bad_input_checks, errors)
        steps.append(
            (repr(fragile_transformer), fragile_transformer)
        )
        if shuffle:
            rand = random.Random()
            rand.seed(seed)
            rand.shuffle(steps)

        super().__init__(steps)


def get_arr_desc(arr):
    desc = '{typ} {shp}'
    typ = type(arr)
    shp = getattr(arr, 'shape', None)
    return desc.format(typ=typ, shp=shp)


class InputLogger(BaseEstimator, TransformerMixin):
    def __init__(self, name=None, level='debug'):
        initialized = False
        if isinstance(level, int):
            self.level = level
            initialized = True
        elif isinstance(level, str):
            level = level.upper()
            if hasattr(logging, level):
                self.level = getattr(logging, level)
                initialized = True

        if not initialized:
            raise ValueError('Invalid level: {}'.format(level))

        self.name = name

    def _log(self, msg):
        # Extract current *.py file name
        # Source: https://stackoverflow.com/a/28645157/2514228
        # TODO get a couple frames up in the stack to get accurate logger name
        if self.name:
            name = self.name
        else:
            name = (inspect.getfile(inspect.currentframe()).split(
                "\\", -1)[-1]).rsplit(".", 1)[0]
        return logging.getLogger(name).log(self.level, msg)

    def fit(self, X, y=None, **fit_kwargs):
        X_desc = get_arr_desc(X)
        y_desc = get_arr_desc(y)
        self._log('Fit called with X={X_desc}, y={y_desc}'.format(
            X_desc=X_desc, y_desc=y_desc))
        return self

    def transform(self, X, **transform_kwargs):
        X_desc = get_arr_desc(X)
        self._log('Transform called with X={X_desc}'.format(X_desc=X_desc))
        return X


def indent(text, n=4):
    _indent = ' ' * n
    return '\n'.join([_indent + line for line in text.split('\n')])


class LoggingContext(object):
    '''
    Logging context manager

    Source: <https://docs.python.org/3/howto/logging-cookbook.html
             #using-a-context-manager-for-selective-logging>
    '''

    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
