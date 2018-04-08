import sklearn.preprocessing

from dengue_prediction.feature.feature import Feature
from dengue_prediction.features.transformers import LagImputer

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
