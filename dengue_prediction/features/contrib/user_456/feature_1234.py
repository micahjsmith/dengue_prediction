import sklearn.preprocessing

from fhub_transformers.nans import LagImputer

input = 'ndvi_ne'
transformer = [
    LagImputer(groupby_kwargs={'level': 'city'}),
    sklearn.preprocessing.Imputer(),
    sklearn.preprocessing.StandardScaler(),
]
