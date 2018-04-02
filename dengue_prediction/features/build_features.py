import logging

import click
import sklearn_pandas

from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.features import (
    get_feature_transformations, get_target_transformations)

logger = logging.getLogger(__name__)


def build_features(X_df):
    logger.info('Building features...')
    feature_transformations = get_feature_transformations()
    mapper = sklearn_pandas.DataFrameMapper([
        t.as_sklearn_pandas_tuple() for t in feature_transformations
    ], input_df=True)
    X = mapper.fit_transform(X_df)
    logger.info('Building features...DONE')
    return X, mapper


def build_target(y_df):
    logger.info('Building target...')
    target_transformations = get_target_transformations()
    mapper = sklearn_pandas.DataFrameMapper([
        t.as_sklearn_pandas_tuple() for t in target_transformations
    ])
    y = mapper.fit_transform(y_df)
    logger.info('Building features...DONE')
    return y, mapper


@click.command()
@click.argument('input_dir', type=click.Path(
    exists=True, readable=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    X_df_tr, y_df_tr = load_data()
    X_tr, mapper_X = build_features(X_df_tr)
    y_tr, mapper_y = build_target(y_df_tr)

    logger.info('Loading data from {}...'.format(input_dir))
    X_df, y_df = load_data(input_dir=input_dir)
    logger.info('Loading data...DONE')

    logger.info('Building features...')
    X = mapper_X.transform(X_df)
    logger.info('Building features...DONE')

    logger.info('Building target...')
    y = mapper_y.transform(y_df)
    logger.info('Building target...DONE')

    return X, y


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    main()
