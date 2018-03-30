import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, value='NaN', replacement=0.0):
        self.value = value
        self.replacement = replacement

    def fit(self, X, y=None, **fit_kwargs):
        return self

    def transform(self, X, **transform_kwargs):
        X = X.copy()
        if self.value != 'NaN':
            mask = X == self.value
        else:
            mask = np.isnan(X)
        X[mask] = self.replacement
        return X
