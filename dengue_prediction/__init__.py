import pathlib
import warnings

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
PROJECT_PATHS = {}
PROJECT_PATHS['root'] = _PROJECT_ROOT
PROJECT_PATHS['data'] = _PROJECT_ROOT / 'data'
PROJECT_PATHS['rawdata'] = _PROJECT_ROOT / 'data' / 'raw'

warnings.filterwarnings(
    'ignore',
    message='Conversion of the second argument of issubdtype',
    module='h5py',
    lineno=36,
)
