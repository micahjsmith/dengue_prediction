import logging
import subprocess
import sys

import click

from dengue_prediction import PROJECT_ROOT
from dengue_prediction.config import cg

logger = logging.getLogger(__name__)


def get_s3_base_url():
    project_name = cg('problem', 'name')
    bucket = cg('data', 's3_bucket')
    return 's3://{bucket}/{project_name}'.format(
        bucket=bucket, project_name=project_name)


def make_aws_command(src, dst, profile=None):
    cmd = ['aws', 's3', 'sync', src, dst]
    if profile is not None:
        cmd.append('--profile')
        cmd.append(profile)
    return cmd


def run_command(cmd):
    output = subprocess.check_output(cmd, universal_newlines=True)
    if output:
        logger.info(output)
    return output


def upload(profile=None):
    base = get_s3_base_url()
    src = str(PROJECT_ROOT.joinpath('data', 'raw'))
    dst = base + '/data/raw'
    cmd = make_aws_command(src, dst, profile=profile)
    return run_command(cmd)


def download(profile=None):
    base = get_s3_base_url()
    src = base + '/data/raw'
    dst = str(PROJECT_ROOT.joinpath('data', 'raw'))
    cmd = make_aws_command(src, dst, profile=profile)
    return run_command(cmd)


@click.command()
@click.argument('direction')
@click.option('--profile', default=None)
def main(direction, profile):
    '''Sync data with s3

    Example usage:
        python -m dengue_prediction.data.sync_data download
    '''
    if direction == 'upload':
        return upload(profile=profile)
    elif direction == 'download':
        return download(profile=profile)
    else:
        raise ValueError(
            'Invalid direction: {}'.format(direction))


if __name__ == '__main__':
    # TODO log only this module
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
