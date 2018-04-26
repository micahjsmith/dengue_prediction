#!/usr/bin/env python

import logging
import sys

import click

import dengue_prediction.features.travis as travis
from dengue_prediction.features.validate_features import validate_by_pr_num

@click.command()
@click.argument('pr_num', required=False, default=None, type=int)
def main(pr_num=None):
    RETVAL_NOT_PR = -1
    RETVAL_VALID = 0
    RETVAL_INVALID = 1

    if pr_num is None:
        pr_num = travis.get_pr_num()
        if pr_num is None:
            return RETVAL_NOT_PR

    result = validate_by_pr_num(pr_num)
    if result is True:
        return RETVAL_VALID
    else:
        return RETVAL_INVALID

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
