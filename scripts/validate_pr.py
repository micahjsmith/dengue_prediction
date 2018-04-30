import logging
import sys

import click
from fhub_core.validation import PullRequestFeatureValidator
from fhub_core.util.travis import get_travis_pr_num

from dengue_prediction.config import cg, load_repo
from dengue_prediction.data.make_dataset import load_data

logger = logging.getLogger(__name__)

@click.command()
@click.argument('pr_num', required=False, default=None, type=int)
def main(pr_num=None):

    RETVAL_VALID = 0
    RETVAL_NOT_PR = 1
    RETVAL_INVALID = 2

    if pr_num is None:
        logger.info('No PR provided. Trying to detect PR from Travis ENV.')
        pr_num = get_travis_pr_num()
        if pr_num is None:
            logger.info('Could not detect PR. Exiting...')
            return RETVAL_NOT_PR
        else:
            logger.info('Detected PR {}'.format(pr_num))

    logger.info('Validating PR {}...'.format(pr_num))

    repo = load_repo()
    comparison_ref = cg('contrib', 'comparison_ref')
    contrib_module_path = cg('contrib', 'module_path')
    X_df, y_df = load_data()
    validator = PullRequestFeatureValidator(
        pr_num, repo, comparison_ref, contrib_module_path, X_df, y_df)
    result = validator.validate()

    if result is True:
        return RETVAL_VALID
    else:
        return RETVAL_INVALID

if __name__ == '__main__':
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    _logger = logging.getLogger('fhub_core.validation')
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(handler)

    sys.exit(main())
