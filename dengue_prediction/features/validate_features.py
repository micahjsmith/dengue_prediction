import importlib
import logging
import pathlib

import git
from fhub_core.contrib import get_contrib_features
from fhub_core.feature import FeatureValidator
from git.exc import GitCommandError

from dengue_prediction.config import load_config, load_repo
from dengue_prediction.data.make_dataset import load_data
from dengue_prediction.exceptions import GitError

logger = logging.getLogger(__name__)

# drivers for validating features in file or list of files


def validate_feature_file(file):
    mod = import_module_from_relpath(file)
    features = get_contrib_features(mod)

    # todo get small subset
    X_df_tr, y_df_tr = load_data()
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


def validate_feature_file_list(file_list):
    overall_result = True
    for file in file_list:
        result = validate_feature_file(file)
        if result is False:
            overall_result = False

    return overall_result

# drivers for evaluating entire project by head name, SHA, or PR num


def validate_by_pr_num(pr_num):
    file_changes = get_file_changes_by_pr_num(pr_num)
    return validate_feature_file_list(file_changes)


def validate_by_ref_name(ref_name):
    file_changes = get_file_changes_by_ref_name(ref_name)
    return validate_feature_file_list(file_changes)


def validate_by_sha(sha):
    file_changes = get_file_changes_by_sha(sha)
    return validate_feature_file_list(file_changes)

# get file changes compared to some master
# - files should be


def get_file_changes_by_pr_num(pr_num):
    # check that pr has been fetched and there is a local copy
    fetch_pr(pr_num)
    get_file_changes_by_ref_name()


def get_file_changes_by_ref_name(ref_name):
    # just check that head is valid
    repo = load_repo()
    if ref_name not in repo.refs:
        raise ValueError('Invalid ref: {ref_name}'.format(ref_name=ref_name))

    return get_file_changes_by_revision(ref_name)


def get_file_changes_by_sha(sha):
    return get_file_changes_by_revision(sha)


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

def get_reference_branch_sha():
    reference_branch = get_reference_branch_ref_name()
    return get_sha_for_ref_name(reference_branch)


def get_reference_branch_ref_name():
    config = load_config()
    reference_branch = config['problem']['reference_branch']
    return reference_branch


def get_sha_for_ref_name(ref_name):
    repo = load_repo()
    heads = repo.heads
    if ref_name in heads:
        return heads[ref_name].commit.hexsha
    else:
        raise ValueError(
            'Ref name {ref_name} not found in refs'
            .format(ref_name=ref_name))


def get_sha_for_pr_num(pr_num):
    try:
        fetchinfo = fetch_pr(pr_num)
        return fetchinfo.commit.hexsha
    except (GitCommandError, GitError):
        raise ValueError('Could not fetch PR #{pr_num}')


def fetch_pr(pr_num):
    config = load_config()
    reference_remote = config['problem']['reference_remote']
    repo = load_repo()
    remote = repo.remote(name=reference_remote)
    # note: without a 'src:dest' refspec, a `FetchInfo` object is returned, but
    # is sufficient to get the sha
    pr_info = PullRequestInfo(pr_num)
    refspec = '{remote}:{local}'.format(
        remote=pr_info.remote_ref_name,
        local=pr_info.local_ref_name)
    fetchinfo = remote.fetch(refspec=refspec)
    if len(fetchinfo) == 1:
        fetchinfo = fetchinfo[0]
        return fetchinfo
    else:
        raise GitError(
            'Unexpectedly returned {n} git.remote.FetchInfo items'
            .format(n=len(fetchinfo)))


class PullRequestInfo:
    def __init__(self, pr_num):
        self.pr_num = pr_num

    @property
    def local_ref_name(self):
        '''Shorthand name of local ref as 'pull/1' '''
        return 'pull/{pr_num}'.format(pr_num=self.pr_num)

    @property
    def local_rev_name(self):
        '''Full name of revision as 'refs/heads/pull/1' '''
        return 'refs/heads/pull/{pr_num}'.format(pr_num=self.pr_num)

    @property
    def remote_ref_name(self):
        '''Full name of remote ref (e.g. on GitHub) as 'refs/pull/1/head' '''
        return 'refs/pull/{pr_num}/head'.format(pr_num=self.pr_num)


def delete_local_copy_of_pr(pr_num):
    repo = load_repo()
    pr_info = PullRequestInfo(pr_num)
    path = pr_info.local_rev_name
    git.refs.Reference.delete(repo, path)

# util for importing modules


def relpath_to_modname(relpath):
    parts = pathlib.Path(relpath).parts
    if parts[-1] == '__init__.py':
        parts = parts[:-1]
    elif parts[-1].endswith('.py'):
        parts = parts[:-1] + [parts[-1].replace('.py', '')]
    else:
        raise ValueError('Cannot convert a non py file to a modname')

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
