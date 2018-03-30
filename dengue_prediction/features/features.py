import logging


from sklearn.pipeline import make_pipeline
import sklearn.preprocessing


from dengue_prediction.features.feature import Feature
from dengue_prediction.features.transformers import ValueReplacer


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