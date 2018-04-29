import importlib
import logging
import pathlib

import funcy
from fhub_core.contrib import get_contrib_features
from fhub_core.feature import FeatureValidator

from dengue_prediction import PROJECT_ROOT
from dengue_prediction.config import cg
from dengue_prediction.exceptions import UnexpectedFileChangeInPullRequestError

logger = logging.getLogger(__name__)


class HeadInfo:
    def __init__(self, repo):
        self.head = repo.head

    @property
    def path(self):
        return self.head.ref.path


class PullRequestFeatureValidator:
    def __init__(self, repo, pr_num, comparison_ref, contrib_module_path,
                 X_df, y_df):
        '''Validate the features introduced in a proposed pull request

        Args:
            pr_num (str): Pull request number
        '''
        self.repo = repo
        self.pr_num = pr_num
        self.comparison_ref = comparison_ref
        self.contrib_module_path = contrib_module_path
        self.X_df = X_df
        self.y_df = y_df

        self.pr_info = PullRequestInfo(self.pr_num)
        self.head_info = HeadInfo(self.repo)

        # may be set by other methods
        self.file_changes = None
        self.file_changes_admissible = None
        self.file_changes_inadmissible = None
        self.features = None

    def collect_file_changes(self):
        logger.info('Collecting file changes...')

        from_rev = self.comparison_ref
        to_rev = self.pr_info.local_rev_name
        self.file_changes = get_file_changes_by_revision(
            self.repo, from_rev, to_rev)

        # log results
        for i, file in enumerate(self.file_changes):
            logger.debug('File {i}: {file}'.format(i=i, file=file))
        logger.info('Collected {} files'.format(len(self.file_changes)))

    def categorize_file_changes(self):
        '''Partition file changes into admissible and inadmissible changes'''
        if self.file_changes is None:
            raise ValueError('File changes have not been collected.')

        logger.info('Categorizing file changes...')

        self.file_changes_admissible = []
        self.file_changes_inadmissible = []

        # admissible:
        # - within contrib subdirectory
        # - is a .py file
        # - TODO: is a .txt file
        # - is an addition
        # inadmissible:
        # - otherwise (wrong directory, wrong filetype, wrong modification
        #   type)
        def within_contrib_subdirectory(file):
            contrib_relpath = self.contrib_module_path
            return pathlib.Path(contrib_relpath) in pathlib.Path(file).parents

        def is_appropriate_filetype(file):
            return file.endswith('.py')

        def is_appropriate_modification_type(modification_type):
            # TODO
            # return modification_type == 'A'
            return True

        is_admissible = funcy.all_fn(
            within_contrib_subdirectory, is_appropriate_filetype,
            is_appropriate_modification_type)

        for file in self.file_changes:
            if is_admissible(file):
                self.file_changes_admissible.append(file)
                logger.debug(
                    'Categorized {file} as ADMISSIBLE'.format(file=file))
            else:
                self.file_changes_inadmissible.append(file)
                logger.debug(
                    'Categorized {file} as INADMISSIBLE'.format(file=file))

        logger.info('Admitted {} files and rejected {} files'.format(
            len(self.file_changes_admissible),
            len(self.file_changes_inadmissible)))

    def collect_features(self):
        if self.file_changes_admissible is None:
            raise ValueError('File changes have not been collected.')

        logger.info('Collecting features...')

        self.features = []
        for file in self.file_changes_admissible:
            try:
                mod = import_module_from_relpath(file)
            except ImportError:
                logger.exception(
                    'Failed to import module from {}'.format(file))
                continue

            features = get_contrib_features(mod)
            self.features.extend(features)

        logger.info('Collected {} features'.format(len(self.features)))

    def validate_features(self, features):
        # get small subset?
        X_df, y_df = subsample_data_for_validation(self.X_df, self.y_df)

        # validate
        feature_validator = FeatureValidator(X_df, y_df)
        overall_result = True
        for feature in features:
            result, failures = feature_validator.validate(feature)
            if result is True:
                logger.info(
                    'Feature is valid: {feature}'.format(feature=feature))
            else:
                logger.info(
                    'Feature is NOT valid: {feature}'.format(feature=feature))
                logger.debug(
                    'Failures in validation: {failures}'
                    .format(failures=failures))
                overall_result = False

        return overall_result

    def validate(self):
        # check that we are *on* this PR's branch
        expected_ref = self.pr_info.local_rev_name
        current_ref = self.head_info.path
        if expected_ref != current_ref:
            raise NotImplementedError(
                'Must validate PR while on that PR\'s branch')

        # collect
        self.collect_file_changes()
        self.categorize_file_changes()
        self.collect_features()

        # validate
        result = self.validate_features(self.features)

        return result


def subsample_data_for_validation(X_df_tr, y_df_tr):
    # TODO
    return X_df_tr, y_df_tr


# for utils

def get_file_changes_by_revision(repo, from_revision, to_revision):
    '''Get file changes between two revisions

    For details on specifying revisions, see

        git help revisions
    '''
    diff_str = '{from_revision}..{to_revision}'.format(
        from_revision=from_revision, to_revision=to_revision)
    return get_file_changes_by_diff_str(repo, diff_str)


def get_file_changes_by_diff_str(repo, diff_str):
    return repo.git.diff(diff_str, name_only=True).split('\n')


# utils

def get_comparison_ref_name():
    return cg('contrib', 'comparison_ref')


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


def modname_to_relpath(modname, add_init=True):
    '''Convert module name to relative path

    Example:
        >>> modname_to_relpath('dengue_prediction.features')
        'dengue_prediction/features/__init__.py'

    '''
    parts = modname.split('.')
    relpath = pathlib.Path('.').joinpath(*parts)
    if PROJECT_ROOT.joinpath(relpath).is_dir():
        if add_init:
            relpath = relpath.joinpath('__init__.py')
    else:
        relpath = relpath + '.py'
    return str(relpath)


def import_module_from_relpath(file):
    modname = relpath_to_modname(file)
    return import_module_from_modname(modname)


def import_module_from_modname(name):
    return importlib.import_module(name)


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
