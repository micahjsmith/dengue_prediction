import logging

import click

from dengue_prediction.io import save_model
from dengue_prediction.models.api import train_model


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
