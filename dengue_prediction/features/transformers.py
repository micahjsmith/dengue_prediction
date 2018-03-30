from sklearn.base import BaseEstimator, TransformerMixin


class ValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, value='NaN', replacement=0.0):
        self.value = value
        self.replacement = replacement
        
    def fit(X, y=None, **fit_kwargs):
        return self
    
    def transform(X, **transform_kwargs):
        X = X.copy()
        if value != 'NaN':
            mask = X == value
        else:
            mask = np.isnan(X)
        X[mask] = replacement
        return X