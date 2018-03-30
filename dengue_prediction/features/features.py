import numpy as np
import sklearn.preprocessing

from dengue_prediction.features.feature import Feature
from dengue_prediction.features.transformers import (LagImputer, NullFiller,
                                                     NullIndicator,
                                                     SimpleFunctionTransformer,
                                                     SingleLagger)


def get_features():
    features = []

    features.append(
        Feature(
            input='ndvi_ne',
            transformer=[
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_nw',
            transformer=[
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_se',
            transformer=[
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_sw',
            transformer=[
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='precipitation_amt_mm',
            transformer=[
                sklearn.preprocessing.Imputer(),
                SimpleFunctionTransformer(np.log1p)
            ]
        )
    )

    features.append(
        Feature(
            input='precipitation_amt_mm',
            transformer=[
                NullIndicator(),
            ]
        )
    )

    features.append(
        Feature(
            input='reanalysis_sat_precip_amt_mm',
            transformer=[
                SingleLagger(1, groupby_kwargs={'level': 'city'}),
                LagImputer(groupby_kwargs={'level': 'city'}),
                NullFiller(replacement=0.0),
            ]
        )
    )

    return features
