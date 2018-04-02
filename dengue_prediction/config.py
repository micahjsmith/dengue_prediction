import pathlib

import funcy
import yaml

from dengue_prediction import PROJECT_PATHS


def get_config_schema():
    # TODO
    return None


def validate_config(config, schema):
    # TODO
    return True


@funcy.memoize
def load_config():
    config_fn = PROJECT_PATHS['root'] / 'config.yml'
    with open(config_fn, 'r') as f:
        config = yaml.load(f)
    schema = get_config_schema()
    if validate_config(config, schema):
        return config
    else:
        # TODO
        raise RuntimeError


def get_table_config(table_name):
    config = load_config()
    result = funcy.select(lambda d: d['name'] == table_name, config['tables'])
    if len(result) == 1:
        return result[0]
    else:
        raise ValueError(
            "Multiple configs found for table name '{}'".format(table_name))


def get_train_dir():
    config = load_config()
    return PROJECT_PATHS['root'].joinpath(config['problem']['data']['train'])


def get_table_abspath(containing_dir, table_name):
    table_config = get_table_config(table_name)
    return pathlib.Path(containing_dir).joinpath(table_config['path'])
