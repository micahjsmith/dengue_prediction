import sklearn.preprocessing

from dengue_prediction.features.feature import Feature


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

    return features
