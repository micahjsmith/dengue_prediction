import pathlib
import warnings

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
PROJECT_PATHS = {}
PROJECT_PATHS['root'] = PROJECT_ROOT
PROJECT_PATHS['data'] = PROJECT_ROOT / 'data'
PROJECT_PATHS['rawdata'] = PROJECT_ROOT / 'data' / 'raw'

warnings.filterwarnings(
    'ignore',
    message='Conversion of the second argument of issubdtype',
    module='h5py',
    lineno=36,
)

warnings.filterwarnings(
    'ignore',
    message='Predicted variances smaller than 0.',
    module='sklearn.gaussian_process.gpr',
    lineno=335,
)
