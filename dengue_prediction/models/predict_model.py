import logging

import click

from dengue_prediction.io import save_predictions
from dengue_prediction.models.api import predict_model


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
