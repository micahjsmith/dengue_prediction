import pathlib

import funcy
import git
import yaml

from dengue_prediction import PROJECT_ROOT
from dengue_prediction.exceptions import ConfigurationError


def get_config_schema():
    # TODO
    return None


def validate_config(config, schema):
    # TODO
    if True:
        return True
    else:
        raise ConfigurationError('Bad config for schema.')


def load_config():
    config_fn = PROJECT_ROOT / 'config.yml'
    with open(str(config_fn), 'r') as f:
        config = yaml.load(f)
    schema = get_config_schema()
    if validate_config(config, schema):
        return config
    else:
        # TODO
        raise RuntimeError


def cg(*args):
    '''Get nested path from config

    Returns:
        Result, or None if path not found
    '''
    config = load_config()
    return funcy.get_in(config, args, default=None)


def load_repo():
    return git.Repo(str(PROJECT_ROOT))


def get_table_config(table_name):
    result = funcy.select(lambda d: d['name'] == table_name, cg('tables'))
    if len(result) == 1:
        return result[0]
    else:
        raise ValueError(
            "Multiple configs found for table name '{}'".format(table_name))


def get_train_dir():
    load_config()
    return PROJECT_ROOT.joinpath(cg('data', 'train'))


def get_table_abspath(containing_dir, table_name):
    table_config = get_table_config(table_name)
    return pathlib.Path(containing_dir).joinpath(table_config['path'])
