# drivers for validating features in file or list of files


def validate_feature_file(filepath):
    pass


def validate_feature_file_list(filepath_list):
    pass

# drivers for evaluating entire project by head name, SHA, or PR num


def validate_by_pr_num(pr_num):
    pass


def validate_by_head_name(head_name):
    pass


def validate_by_sha(sha):
    pass

# get file changes compared to some master


def get_file_changes_by_pr_num(pr_num):
    pass


def get_file_changes_by_head_name(head_name):
    pass


def get_file_changes_by_sha(sha):
    pass

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
