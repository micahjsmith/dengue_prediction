import sklearn.preprocessing
from fhub_core.feature import Feature
from fhub_transformers.missing import LagImputer

features = [
    Feature(
        input='ndvi_nw',
        transformer=[
            LagImputer(groupby_kwargs={'level': 'city'}),
            sklearn.preprocessing.Imputer(),
            sklearn.preprocessing.StandardScaler(),
        ],
    ),
]
