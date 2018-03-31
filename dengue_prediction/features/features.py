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
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_nw',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_se',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='ndvi_sw',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
                sklearn.preprocessing.Imputer(),
                sklearn.preprocessing.StandardScaler(),
            ]
        )
    )

    features.append(
        Feature(
            input='precipitation_amt_mm',
            transformer=[
                LagImputer(groupby_kwargs={'level': 'city'}),
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

    # Same basic cleaning of time series features.
    for input_ in [
        'reanalysis_air_temp_k',
        'reanalysis_avg_temp_k',
        'reanalysis_dew_point_temp_k',
        'reanalysis_max_air_temp_k',
        'reanalysis_min_air_temp_k',
        'reanalysis_precip_amt_kg_per_m2',
        'reanalysis_relative_humidity_percent',
        'reanalysis_specific_humidity_g_per_kg',
        'reanalysis_tdtr_k',

        'station_avg_temp_c',
        'station_diur_temp_rng_c',
        'station_max_temp_c',
        'station_min_temp_c',
        'station_precip_mm',
    ]:
        features.append(
            Feature(
                input=input_,
                transformer=[
                    LagImputer(groupby_kwargs={'level': 'city'}),
                    NullFiller(replacement=0.0),
                    sklearn.preprocessing.StandardScaler(),
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
