import logging


import sklearn_pandas


from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.features import get_features


logger = logging.getLogger(__name__)


def build_features():
    X_df, y = load_data()

    logger.info('Building features...')
    features = get_features()
    mapper = sklearn_pandas.DataFrameMapper([
        feature.as_sklearn_pandas_tuple() for feature in features
    ], input_df=True)
    X = mapper.fit_transform(X_df, y)
    logger.info('Building features...DONE')
    return X


if __name__ == '__main__':
    build_features()
