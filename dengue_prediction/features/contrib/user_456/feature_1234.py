import sklearn.preprocessing

from dengue_prediction.features.transformers import LagImputer

input = 'ndvi_ne'
transformer = [
    LagImputer(groupby_kwargs={'level': 'city'}),
    sklearn.preprocessing.Imputer(),
    sklearn.preprocessing.StandardScaler(),
]
