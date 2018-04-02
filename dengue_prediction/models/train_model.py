import logging
import os
import pathlib

import click

from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.features.build_features import (
    build_features, build_target)
from dengue_prediction.models.modeler import create_model

logger = logging.getLogger(__name__)


def train_model():
    logger.info('Training model...')
    X_df_tr, y_df_tr = load_data()
    X_tr, mapper_X = build_features(X_df_tr)
    y_tr, mapper_y = build_target(y_df_tr)
    model = create_model()
    model.fit(X_tr, y_df_tr)
    logger.info('Training model...DONE')
    return model


def save_model(model, output_dir):
    logger.info('Saving model...')
    os.makedirs(output_dir, exist_ok=True)
    filepath = pathlib.Path(output_dir).joinpath('model.pkl')
    model.dump(filepath)
    logger.info('Saving model...DONE ({})'.format(filepath))


@click.command()
@click.argument('output_dir', type=click.Path())
def main(output_dir):
    '''Train model and save parameters to output_dir

    Example usage:
        python -m dengue_prediction.models.train_model /path/to/output/dir
    '''
    model = train_model()
    save_model(model, output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
