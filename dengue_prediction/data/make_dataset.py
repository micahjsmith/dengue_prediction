import logging

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

import dengue_prediction.config


def load_data(input_dir=None):
    if input_dir is None:
        return _load_data_using_config()
    else:
        return _load_data_from_dir(input_dir)


def _load_data_using_config():
    train_dir = dengue_prediction.config.get_train_dir()
    return _load_data_from_dir(train_dir)


def _load_data_from_dir(input_dir):
    X = _load_entities_table(input_dir)
    y = _load_target_table(input_dir)
    return X, y


def _load_entities_table(input_dir):
    return _load_table_type('entities_table_name', input_dir)


def _load_target_table(input_dir):
    return _load_table_type('target_table_name', input_dir)


def _load_table_type(table_type, input_dir):
    config = dengue_prediction.config.load_config()
    return _load_named_table(
        input_dir, config['problem']['data'][table_type])


def _load_named_table(input_dir, table_name):
    table_config = dengue_prediction.config.get_table_config(table_name)
    pd_read_kwargs = table_config['pd_read_kwargs']
    fn = dengue_prediction.config.get_table_abspath(input_dir, table_name)
    return pd.read_csv(fn, **pd_read_kwargs)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    ''' Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    '''
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
