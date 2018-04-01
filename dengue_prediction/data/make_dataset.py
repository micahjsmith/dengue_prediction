import logging
import os

import click
import funcy
import pandas as pd
from dotenv import find_dotenv, load_dotenv

import dengue_prediction.config


def load_data():
    X = load_entities_table()
    y = load_target_table()
    return X, y


def load_entities_table():
    config = dengue_prediction.config.load_config()
    return _load_named_table(config['problem']['data']['entities_table_name'])


def load_target_table():
    config = dengue_prediction.config.load_config()
    return _load_named_table(config['problem']['data']['target_table_name'])


def _load_named_table(table_name):
    table_config = dengue_prediction.config.get_table_config(table_name)
    fn = dengue_prediction.config.get_table_abspath(table_name)
    kwargs = table_config['pd_read_kwargs']
    with open(fn, 'r') as f:
        return pd.read_csv(f, **kwargs)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
