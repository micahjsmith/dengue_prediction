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


def run_aws_s3_sync(src, dst, credentials=True, profile=None):
    cmd = ['aws', 's3', 'sync']
    if not credentials:
        cmd.insert(1, '--no-sign-request')
        region = cg('data', 's3_bucket_region')
        cmd.append('--region={}'.format(region))
        cmd.append('--source-region={}'.format(region))
    cmd.append(src)
    cmd.append(dst)
    if profile is not None:
        cmd.append('--profile')
        cmd.append(profile)
    try:
        logger.debug('Executing command: {}'.format(' '.join(cmd)))
        output = subprocess.check_output(
            cmd, universal_newlines=True, stderr=subprocess.STDOUT)
        if output:
            logger.info(output)
        return output
    except subprocess.CalledProcessError as e:
        if 'Unable to locate credentials' in e.output and credentials:
            return run_aws_s3_sync(
                src, dst, credentials=False, profile=profile)
        else:
            raise


def upload(profile=None):
    base = get_s3_base_url()
    src = str(PROJECT_ROOT.joinpath('data', 'raw'))
    dst = base + '/data/raw'
    return run_aws_s3_sync(src, dst, profile=profile)


def download(profile=None):
    base = get_s3_base_url()
    src = base + '/data/raw'
    dst = str(PROJECT_ROOT.joinpath('data', 'raw'))
    return run_aws_s3_sync(src, dst, profile=profile)


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
