import random

import funcy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pandas.pipeline import TransformerPipeline

EPSILON = 1e-4


class _IdentityTransformer(BaseEstimator, TransformerMixin):
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
            ('IdentityTransformer{:02d}'.format(i), _IdentityTransformer())
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


@funcy.contextmanager
def seeded(seed):
    '''Set seed, run code, then restore rng state'''
    if seed is not None:
        np_random_state = np.random.get_state()
        random_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)

    yield

    if seed is not None:
        np.random.set_state(np_random_state)
        random.setstate(random_state)


@funcy.contextmanager
def log_seed_on_error(logger, seed=None):
    '''Store seed, run code, and report seed if error'''
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    try:
        with seeded(seed):
            yield
    except Exception as e:
        logger.exception('Error was thrown using seed {}'.format(seed))


def check_frame_equal(*args, **kwargs):
    try:
        pd.testing.assert_frame_equal(*args, **kwargs)
        return True
    except AssertionError:
        return False
