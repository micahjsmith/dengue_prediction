import logging
import os

import click
import sklearn_pandas

from dengue_prediction.config import get_table_abspath, load_config
from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.features import (
    get_feature_transformations, get_target_transformations)
from dengue_prediction.io import write_tabular
from dengue_prediction.util import replaceext, spliceext

logger = logging.getLogger(__name__)


def make_mapper_from_features(features):
    return sklearn_pandas.DataFrameMapper([
        t.as_input_transformer_tuple() for t in features
    ], input_df=True)


def build_features(X_df):
    logger.info('Building features...')
    features = get_feature_transformations()
    mapper = make_mapper_from_features(features)
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
    logger.info('Building target...DONE')
    return y, mapper


def build_features_from_dir(input_dir, return_mappers=False):
    # TODO allow specification of input dir for train data
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

    if return_mappers:
        return X, y, mapper_X, mapper_y
    else:
        return X, y


def save_features(X, y, output_dir):
    logger.info('Saving features...')
    # save to output_dir
    os.makedirs(output_dir, exist_ok=True)

    def namer(table_name, filetype='pkl'):
        fn = get_table_abspath(output_dir, table_name)
        fn = spliceext(fn, '_featurized')
        fn = replaceext(fn, '.' + filetype)
        return fn

    config = load_config()
    entities_table_name = config['problem']['data']['entities_table_name']
    fn_X = namer(entities_table_name)
    write_tabular(X, fn_X)
    logger.info('Saved featurized entities to {}'.format(fn_X))

    target_table_name = config['problem']['data']['target_table_name']
    fn_y = namer(target_table_name)
    write_tabular(y, fn_y)
    logger.info('Saved transformed target to {}'.format(fn_y))


@click.command()
@click.argument('input_dir', type=click.Path(
    exists=True, readable=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    '''Build features from raw data in input_dir and save to output_dir

    Example usage:
        python -m dengue_prediction.features.build_features  \
            /path/to/input/dir /path/to/output/dir
    '''
    X, y = build_features_from_dir(input_dir)
    save_features(X, y, output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
