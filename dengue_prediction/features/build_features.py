import logging

import sklearn_pandas

from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.features import get_features

logger = logging.getLogger(__name__)


def build_features(X_df, y_df):
    logger.info('Building features...')
    feature_transformations = get_feature_transformations()
    mapper = sklearn_pandas.DataFrameMapper([
        t.as_sklearn_pandas_tuple() for t in feature_transformations
    ], input_df=True)
    X = mapper.fit_transform(X_df, y_df)
    logger.info('Building features...DONE')
    return X, mapper


if __name__ == '__main__':
    X_df, y_df = load_data()
    build_features(X_df, y_df)
