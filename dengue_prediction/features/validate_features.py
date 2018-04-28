import importlib
import logging
import pathlib

import git
from fhub_core.contrib import get_contrib_features
from fhub_core.feature import FeatureValidator
from git.exc import GitCommandError

from dengue_prediction.config import load_config, load_repo
from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.exceptions import GitError, UnexpectedFileChangeInPullRequestError

logger = logging.getLogger(__name__)


def subsample_data_for_validation(X_df_tr, y_df_tr):
    # TODO
    return X_df_tr, y_df_tr


def validate_features(features):
    # get small subset
    X_df_tr, y_df_tr = load_data()
    X_df_tr, y_df_tr = subsample_data_for_validation(X_df_tr, y_df_tr)

    # validate
    validator = FeatureValidator(X_df_tr, y_df_tr)
    overall_result = True
    for feature in features:
        result, failures = validator.validate(feature)
        if result is True:
            logger.info('Feature is valid: {feature}'.format(feature=feature))
        else:
            logger.info('Feature is NOT valid: {feature}'
                        .format(feature=feature))
            logger.debug('Failures in validation: {failures}'
                         .format(failures=failures))
            overall_result = False

    return overall_result


def validate_feature_file(file):
    try:
        logger.info('Attempting to validate changes in {file}'
                    .format(file=file))
        mod = import_module_from_relpath(file)
    except UnexpectedFileChangeInPullRequestError:
        # TODO mark failure
        return False

    features = get_contrib_features(mod)
    return validate_features(features)


def validate_feature_file_list(file_list):
    overall_result = True
    for file in file_list:
        result = validate_feature_file(file)
        if result is False:
            overall_result = False

    return overall_result

def validate_by_pr_num(pr_num):
    file_changes = get_file_changes_by_pr_num(pr_num)
    return validate_feature_file_list(file_changes)


def get_file_changes_by_pr_num(pr_num):
    pr_info = PullRequestInfo(pr_num)
    return get_file_changes_by_ref_name(pr_info.local_ref_name)


def get_file_changes_by_ref_name(ref_name):
    # just check that head is valid
    repo = load_repo()
    if ref_name not in repo.refs:
        raise ValueError('Invalid ref: {ref_name}'.format(ref_name=ref_name))

    return get_file_changes_by_revision(ref_name)


def get_file_changes_by_revision(revision):
    '''Get file changes between reference branch and specified revision

    For details on specifying revisions, see

        git help revisions
    '''
    reference_ref_name = get_reference_branch_ref_name()
    diff_str = '{from_}..{to}'.format(
        from_=reference_ref_name, to=revision)
    return get_file_changes_by_diff_str(diff_str)


def get_file_changes_by_diff_str(diff_str):
    repo = load_repo()
    return repo.git.diff(diff_str, name_only=True).split('\n')


# utils

def get_reference_branch_ref_name():
    config = load_config()
    reference_branch = config['problem']['reference_branch']
    return reference_branch


class PullRequestInfo:
    def __init__(self, pr_num):
        self.pr_num = pr_num

    def _format(self, str):
        return str.format(pr_num=self.pr_num)

    @property
    def local_ref_name(self):
        '''Shorthand name of local ref, e.g. 'pull/1' '''
        return self._format('pull/{pr_num}')

    @property
    def local_rev_name(self):
        '''Full name of revision, e.g. 'refs/heads/pull/1' '''
        return self._format('refs/heads/pull/{pr_num}')

    @property
    def remote_ref_name(self):
        '''Full name of remote ref (as on GitHub), e.g. 'refs/pull/1/head' '''
        return self._format('refs/pull/{pr_num}/head')


# util for importing modules


def relpath_to_modname(relpath):
    parts = pathlib.Path(relpath).parts
    if parts[-1] == '__init__.py':
        parts = parts[:-1]
    elif parts[-1].endswith('.py'):
        parts = list(parts)
        parts[-1] = parts[-1].replace('.py', '')
    else:
        msg = 'Cannot convert a non-python file to a modname'
        msg_detail = 'The relpath given is: {}'.format(relpath)
        logger.error(msg + '\n' + msg_detail)
        raise UnexpectedFileChangeInPullRequestError(msg)

    return '.'.join(parts)


def import_module_from_relpath(file):
    modname = relpath_to_modname(file)
    mod = importlib.import_module(modname)
    return mod


# producing nice output after validation is complete


class FeatureValidationReport:
    TEMPLATE = \
        '''
    this is the template
    '''

    def __init__(self):
        self.env = None

    def render(self):
        pass
