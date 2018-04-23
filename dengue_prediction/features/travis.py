import logging
import os

logger = logging.getLogger(__name__)


def get_pr_num():
    # The pull request number if the current job is a pull request, “false” if
    # it’s not a pull request.
    # See  https://docs.travis-ci.com/user/environment-variables/#Default-Environment-Variables  # noqa
    if 'TRAVIS_PULL_REQUEST' in os.environ:
        travis_pull_request = os.environ['TRAVIS_PULL_REQUEST']
        if travis_pull_request == 'false':
            return None
        else:
            try:
                pr_num = int(travis_pull_request)
                return pr_num
            except ValueError:
                return None
    else:
        return None


def is_pr():
    return get_pr_num() is not None


def get_pr_sha():
    # if the current job is a pull request, the commit SHA of the HEAD commit
    # of the PR.
    # if the current job is a push build, this variable is empty ("").
    # See  https://docs.travis-ci.com/user/environment-variables/#Default-Environment-Variables  # noqa
    if 'TRAVIS_PULL_REQUEST_SHA' in os.environ:
        travis_pull_request_sha = os.environ['TRAVIS_PULL_REQUEST_SHA']
        if travis_pull_request_sha == '':
            return None
        else:
            return travis_pull_request_sha
    else:
        return None
