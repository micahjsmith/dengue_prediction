import logging

import click

from dengue_prediction.models.api import evaluate_model


@click.command()
@click.argument('input_dir', required=False, default=None, type=click.Path(
    exists=True, readable=True, dir_okay=True))
def main(input_dir):
    '''Train model, then evaluate on input_dir. If input_dir is omitted,
    computes cross-validated metrics on training data.

    Example usage:
        python -m dengue_prediction.models.evaluate_model \
            /path/to/input/dir
    '''
    results = evaluate_model(train_dir=None, test_dir=input_dir)
    print(results)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
