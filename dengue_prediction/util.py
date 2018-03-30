import random

import funcy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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