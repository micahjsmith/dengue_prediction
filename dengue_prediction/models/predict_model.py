import logging
import os
import pathlib

import click

from dengue_prediction.features.build_features import build_features_from_dir
from dengue_prediction.io import write_tabular
from dengue_prediction.models.train_model import train_model

logger = logging.getLogger(__name__)


def _predict_model(model, X_te, y_te):
    y_te_pred = model.predict(X_te)
    # TODO add inverse transform
    # y_te_pred = mapper_y.inverse_transform(y_te_pred)
    return y_te_pred


def predict_model(input_dir):
    model = train_model()
    X_te, y_te = build_features_from_dir(input_dir)
    y_te_pred = _predict_model(model, X_te, y_te)
    return y_te_pred


def save_predictions(y, output_dir):
    logger.info('Saving predictions...')
    os.makedirs(output_dir, exist_ok=True)
    filepath = pathlib.Path(output_dir).joinpath('predictions.pkl')
    write_tabular(y, filepath)
    logger.info('Saving predictions...DONE ({})'.format(filepath))


@click.command()
@click.argument('input_dir', type=click.Path(
    exists=True, readable=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    '''Make predictions on raw data in input_dir and save to output_dir

    Example usage:
        python -m dengue_prediction.models.predict_model \
            /path/to/input/dir /path/to/output/dir
    '''
    y_te_pred = predict_model(input_dir)
    save_predictions(y_te_pred, output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
