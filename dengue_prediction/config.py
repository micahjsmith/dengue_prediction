import funcy
import yaml

from dengue_prediction import PROJECT_PATHS


def validate_config(config, schema):
    return True


@funcy.memoize
def load_config():
    config_fn = PROJECT_PATHS['root'] / 'config.yml'
    with open(config_fn, 'r') as f:
        config = yaml.load(f)
    return config


def get_table_config(table_name):
    config = load_config()
    result = funcy.select(lambda d: d['name'] == table_name, config['tables'])
    if len(result) == 1:
        return result[0]
    else:
        raise ValueError(
            "Multiple configs found for table name '{}'".format(table_name))


def get_table_abspath(table_name):
    config = load_config()
    table_config = get_table_config(table_name)
    fn = (PROJECT_PATHS['root']
          .joinpath(config['problem']['data']['train'])
          .joinpath(table_config['path'])
          )
    return fn
